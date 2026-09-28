<?php

use Illuminate\Support\Facades\Config;
use Illuminate\Support\Facades\Http;
use Laravel\Ai\Ai;
use Laravel\Ai\Approvals\Decisions;
use Laravel\Ai\Classification;
use Laravel\Ai\Enums\Lab;
use Laravel\Ai\Messages\AssistantMessage;
use Laravel\Ai\Messages\Message;
use Laravel\Ai\Messages\UserMessage;
use Laravel\Ai\Prompts\ClassificationPrompt;
use Laravel\Ai\Providers\Provider;
use Laravel\Ai\Providers\Tools\ToolSearch;
use Laravel\Ai\Responses\Data\BooleanAnswer;
use Laravel\Ai\Responses\Data\ChoiceAnswer;
use Laravel\Ai\Responses\Data\ToolCall;
use Laravel\Ai\Tools\ToolNameResolver;
use Laravel\Ai\Tools\ToolSelectionContext;
use Laravel\Ai\Tools\ToolSelector;
use Tests\Fixtures\Agents\RememberingToolSelectorAgent;
use Tests\Fixtures\Agents\ToolSelectorAgent;
use Tests\Fixtures\Tools\ApprovableNumberGenerator;
use Tests\Fixtures\Tools\FixedNumberGenerator;
use Tests\Fixtures\Tools\NamedTool;
use Tests\Fixtures\Tools\RandomNumberGenerator;

test('selects one tool by classifying the prompt', function (): void {
    Classification::fake([
        ['applies' => new BooleanAnswer(1.0), 'tool' => new ChoiceAnswer('custom_named_tool', [
            'custom_named_tool' => 0.9,
            'FixedNumberGenerator' => 0.1,
        ], 0.9)],
    ]);

    $agent = new ToolSelectorAgent(
        (new ToolSelector([
            new NamedTool,
            new FixedNumberGenerator,
        ], 'Select the best tool for this request.'))
            ->using(Lab::OpenRouter, 'openai/gpt-5-mini')
            ->timeout(10)
            ->withProviderOptions(['temperature' => 0]),
    );

    Ai::fakeAgent($agent::class, [
        new ToolCall('call_1', 'custom_named_tool', []),
        'Done.',
    ]);

    $response = $agent->prompt('Refund this order');

    expect($response->toolResults->first()->result)->toBe('ok');

    Classification::assertClassified(fn (ClassificationPrompt $prompt): bool => $prompt->state === 'Refund this order'
        && $prompt->provider->name() === Lab::OpenRouter->value
        && $prompt->model === 'openai/gpt-5-mini'
        && $prompt->timeout === 10
        && $prompt->providerOptions === ['temperature' => 0]
        && $prompt->questions['tool']->toArray() === [
            'type' => 'choice',
            'instructions' => 'Select the best tool for this request.',
            'options' => [
                'custom_named_tool' => 'A tool that declares its own name.',
                'FixedNumberGenerator' => 'This tool can be used to generate cryptographically secure random numbers.',
            ],
        ]
        && $prompt->questions['applies']->toArray() === [
            'type' => 'boolean',
            'instructions' => 'Does the user need one of the listed actions to be carried out?',
            'criteria' => [
                'true' => 'The user needs one of these actions to be carried out: A tool that declares its own name. This tool can be used to generate cryptographically secure random numbers.',
                'false' => 'The user wants only an answer, advice, an explanation or a draft, and no action is carried out: A tool that declares its own name. This tool can be used to generate cryptographically secure random numbers.',
            ],
        ]);
});

test('exposes no tool when the request needs none of them', function (): void {
    Classification::fake([
        [
            'applies' => new BooleanAnswer(0.2),
            'tool' => new ChoiceAnswer('custom_named_tool', [
                'custom_named_tool' => 0.9,
                'FixedNumberGenerator' => 0.1,
            ], 0.8),
        ],
    ]);

    $agent = new ToolSelectorAgent;

    Ai::fakeAgent($agent::class, ['It will be sunny.']);

    $agent->prompt('What is the weather in Geneva tomorrow?');

    expect($agent->stepTools)->toBe([[]]);
});

test('classifier provider options do not become agent provider options', function (): void {
    $classificationProvider = null;

    Classification::fake([
        ['applies' => new BooleanAnswer(1.0), 'tool' => new ChoiceAnswer('custom_named_tool', [
            'custom_named_tool' => 1.0,
            'FixedNumberGenerator' => 0.0,
        ], 1.0)],
    ]);

    $agent = new ToolSelectorAgent(
        (new ToolSelector([new NamedTool, new FixedNumberGenerator]))
            ->withProviderOptions(function (Provider $provider) use (&$classificationProvider): array {
                $classificationProvider = $provider;

                return ['temperature' => 0];
            }),
    );

    Ai::fakeAgent($agent::class, ['Done.']);

    $agent->prompt('Find my order');

    expect($classificationProvider->name())->toBe(Lab::TypeSafe->value)
        ->and($agent->providerOptions)->toBe([null]);
});

test('selects a tool once before streaming starts', function (): void {
    $classifications = 0;

    Classification::fake(function () use (&$classifications): array {
        $classifications++;

        return ['applies' => new BooleanAnswer(1.0), 'tool' => new ChoiceAnswer('custom_named_tool', [
            'custom_named_tool' => 1.0,
            'FixedNumberGenerator' => 0.0,
        ], 1.0)];
    });

    ToolSelectorAgent::fake(['Done.']);

    (new ToolSelectorAgent)->stream('Find my order')->each(fn (): bool => true);

    expect($classifications)->toBe(1);
});

test('selects a tool with the most recent conversation turn', function (): void {
    Classification::fake([
        ['applies' => new BooleanAnswer(1.0), 'tool' => new ChoiceAnswer('custom_named_tool', [
            'custom_named_tool' => 1.0,
            'FixedNumberGenerator' => 0.0,
        ], 1.0)],
    ]);

    $agent = new ToolSelectorAgent(
        (new ToolSelector([new NamedTool, new FixedNumberGenerator]))
            ->withConversationContext(),
    );

    Ai::fakeAgent($agent::class, ['Done.']);

    $agent
        ->withMessages([
            new UserMessage('Find invoice 100.'),
            new AssistantMessage('I found it.'),
            new Message('tool_result', 'internal tool output'),
            new UserMessage('Cancel the related subscription.'),
            new AssistantMessage('Do you want me to continue?'),
        ])
        ->prompt('Yes, do that.');

    Classification::assertClassified(fn (ClassificationPrompt $prompt): bool => $prompt->state === [
        'current_prompt' => 'Yes, do that.',
        'conversation' => [
            ['role' => 'user', 'content' => 'Cancel the related subscription.'],
            ['role' => 'assistant', 'content' => 'Do you want me to continue?'],
        ],
    ]);
});

test('selects a tool with custom state instead of conversation context', function (): void {
    Classification::fake([
        ['applies' => new BooleanAnswer(1.0), 'tool' => new ChoiceAnswer('custom_named_tool', [
            'custom_named_tool' => 1.0,
            'FixedNumberGenerator' => 0.0,
        ], 1.0)],
    ]);

    $agent = new ToolSelectorAgent(
        (new ToolSelector([new NamedTool, new FixedNumberGenerator]))
            ->withConversationContext(2)
            ->withState(fn (ToolSelectionContext $context): array => [
                'request' => $context->prompt,
                'account_status' => 'active',
            ]),
    );

    Ai::fakeAgent($agent::class, ['Done.']);

    $agent
        ->withMessages([new UserMessage('Cancel my subscription.')])
        ->prompt('Yes, continue.');

    Classification::assertClassified(fn (ClassificationPrompt $prompt): bool => $prompt->state === [
        'request' => 'Yes, continue.',
        'account_status' => 'active',
    ]);
});

test('a run resumed after approval keeps the tool that awaits approval', function (): void {
    Config::set('ai.conversations.generate_title', false);

    $states = [];

    Classification::fake(function (ClassificationPrompt $prompt) use (&$states): array {
        $states[] = $prompt->state;

        return [
            'applies' => new BooleanAnswer(1.0),
            'tool' => count($states) === 1
                ? new ChoiceAnswer('ApprovableNumberGenerator', ['ApprovableNumberGenerator' => 1.0, 'custom_named_tool' => 0.0], 1.0)
                : new ChoiceAnswer('custom_named_tool', ['ApprovableNumberGenerator' => 0.0, 'custom_named_tool' => 1.0], 1.0),
        ];
    });

    Http::fake([
        'api.anthropic.com/*' => Http::sequence([
            Http::response([
                'id' => 'msg_1',
                'type' => 'message',
                'role' => 'assistant',
                'model' => 'claude-sonnet-4-6',
                'content' => [['type' => 'tool_use', 'id' => 'toolu_1', 'name' => 'ApprovableNumberGenerator', 'input' => (object) []]],
                'stop_reason' => 'tool_use',
                'usage' => ['input_tokens' => 10, 'output_tokens' => 5],
            ]),
            Http::response([
                'id' => 'msg_2',
                'type' => 'message',
                'role' => 'assistant',
                'model' => 'claude-sonnet-4-6',
                'content' => [['type' => 'text', 'text' => 'The number is 72019.']],
                'stop_reason' => 'end_turn',
                'usage' => ['input_tokens' => 10, 'output_tokens' => 5],
            ]),
        ]),
    ]);

    $user = (object) ['id' => 1];

    $paused = (new RememberingToolSelectorAgent)->forUser($user)->prompt('Generate a number', provider: 'anthropic');

    $resumed = (new RememberingToolSelectorAgent)
        ->continue($paused->conversationId, $user)
        ->prompt(Decisions::from(['toolu_1' => true]), provider: 'anthropic');

    expect($resumed->toolResults[0]->result)->toBe('72019')
        ->and($states)->toBe(['Generate a number']);
});

test('rejects an invalid tool list', function (array $tools, string $message): void {
    expect(fn () => new ToolSelector($tools))->toThrow(InvalidArgumentException::class, $message);
})->with([
    'one tool' => [[new NamedTool], 'A tool selector requires at least two tools.'],
    'not a tool' => [[new NamedTool, new ToolSelectorAgent], 'Tool selectors may only contain tools.'],
    'duplicate names' => [[new NamedTool, new NamedTool], 'Tool selector names must be unique.'],
]);

test('rejects conversation context outside one to five turns', function (int $turns): void {
    expect(fn () => (new ToolSelector([new NamedTool, new FixedNumberGenerator]))->withConversationContext($turns))
        ->toThrow(InvalidArgumentException::class, 'Conversation context must contain between one and five turns.');
})->with([0, 6]);

test('conversation context keeps the requested number of user turns', function (): void {
    $context = new ToolSelectionContext(new ToolSelectorAgent, 'Do it.', [
        new UserMessage('First.'),
        new AssistantMessage('One.'),
        new AssistantMessage(''),
        new UserMessage('Second.'),
        new AssistantMessage('Two.'),
        new UserMessage('Third.'),
    ]);

    expect($context->stateWithConversation(2)['conversation'])->toBe([
        ['role' => 'user', 'content' => 'Second.'],
        ['role' => 'assistant', 'content' => 'Two.'],
        ['role' => 'user', 'content' => 'Third.'],
    ])->and($context->stateWithConversation(5)['conversation'])->toHaveCount(5);
});

test('conversation context is empty when the history has no user message', function (): void {
    $context = new ToolSelectionContext(new ToolSelectorAgent, 'Hello.', [new AssistantMessage('Welcome.')]);

    expect($context->stateWithConversation(1))->toBe(['current_prompt' => 'Hello.', 'conversation' => []]);
});

test('conversation context replaces an earlier custom state', function (): void {
    Classification::fake([
        ['applies' => new BooleanAnswer(1.0), 'tool' => new ChoiceAnswer('custom_named_tool', [], 1.0)],
    ]);

    (new ToolSelector([new NamedTool, new FixedNumberGenerator]))
        ->withState(fn (): string => 'custom state')
        ->withConversationContext()
        ->select(new ToolSelectionContext(new ToolSelectorAgent, 'Go.', []));

    Classification::assertClassified(fn (ClassificationPrompt $prompt): bool => $prompt->state === [
        'current_prompt' => 'Go.',
        'conversation' => [],
    ]);
});

test('selects the tool when the gate is exactly one half', function (): void {
    Classification::fake([
        ['applies' => new BooleanAnswer(0.5), 'tool' => new ChoiceAnswer('custom_named_tool', [], 1.0)],
    ]);

    expect((new ToolSelector([new NamedTool, new FixedNumberGenerator]))->select('Go.'))->toBeInstanceOf(NamedTool::class);
});

test('fails when the classification answers with the wrong types', function (): void {
    Classification::fake([
        ['applies' => new ChoiceAnswer('custom_named_tool', [], 1.0), 'tool' => new BooleanAnswer(1.0)],
    ]);

    (new ToolSelector([new NamedTool, new FixedNumberGenerator]))->select('Go.');
})->throws(LogicException::class, 'Tool selection requires a boolean and a choice answer.');

test('fails when the classification selects an unknown tool', function (): void {
    Classification::fake([
        ['applies' => new BooleanAnswer(1.0), 'tool' => new ChoiceAnswer('missing_tool', [], 1.0)],
    ]);

    (new ToolSelector([new NamedTool, new FixedNumberGenerator]))->select('Go.');
})->throws(InvalidArgumentException::class, 'Classification selected an unknown tool [missing_tool].');

test('an empty selection leaves the other tools in place', function (): void {
    Classification::fake([
        ['applies' => new BooleanAnswer(0.0), 'tool' => new ChoiceAnswer('custom_named_tool', [], 1.0)],
    ]);

    $agent = new ToolSelectorAgent;
    $numbers = new RandomNumberGenerator;

    Ai::fakeAgent($agent::class, ['Done.']);

    $agent->withTools(fn (array $tools): array => [$numbers, ...$tools])->prompt('Hello.');

    expect($agent->stepTools)->toBe([[$numbers]]);
});

test('a selector inside tool search is resolved and dropped when empty', function (float $applies, array $expected): void {
    Classification::fake([
        ['applies' => new BooleanAnswer($applies), 'tool' => new ChoiceAnswer('custom_named_tool', [], 1.0)],
    ]);

    $agent = new ToolSelectorAgent;

    Ai::fakeAgent($agent::class, ['Done.']);

    $agent->withTools(fn (array $tools): array => [new ToolSearch([new RandomNumberGenerator, ...$tools])])->prompt('Hello.');

    expect(array_map(ToolNameResolver::resolve(...), $agent->stepTools[0][0]->tools))->toBe($expected);
})->with([
    'selected' => [1.0, ['RandomNumberGenerator', 'custom_named_tool']],
    'empty' => [0.0, ['RandomNumberGenerator']],
]);

test('a selector keeps its closures after serialization', function (): void {
    Classification::fake([
        ['applies' => new BooleanAnswer(1.0), 'tool' => new ChoiceAnswer('custom_named_tool', [], 1.0)],
    ]);

    $selector = unserialize(serialize(
        (new ToolSelector([new NamedTool, new FixedNumberGenerator]))
            ->withProviderOptions(fn (): array => ['temperature' => 0])
            ->withState(fn (ToolSelectionContext $context): array => ['request' => $context->prompt]),
    ));

    $selector->select(new ToolSelectionContext(new ToolSelectorAgent, 'Go.', []));

    Classification::assertClassified(fn (ClassificationPrompt $prompt): bool => $prompt->state === ['request' => 'Go.']
        && $prompt->providerOptions === ['temperature' => 0]);
});

test('a streamed run resumed after approval keeps the tool that awaits approval', function (): void {
    Config::set('ai.conversations.generate_title', false);

    $classifications = 0;

    Classification::fake(function () use (&$classifications): array {
        $classifications++;

        return [
            'applies' => new BooleanAnswer(1.0),
            'tool' => new ChoiceAnswer('ApprovableNumberGenerator', ['ApprovableNumberGenerator' => 1.0, 'custom_named_tool' => 0.0], 1.0),
        ];
    });

    Http::fake([
        'api.anthropic.com/*' => Http::sequence([
            Http::response([
                'id' => 'msg_1',
                'type' => 'message',
                'role' => 'assistant',
                'model' => 'claude-sonnet-4-6',
                'content' => [['type' => 'tool_use', 'id' => 'toolu_1', 'name' => 'ApprovableNumberGenerator', 'input' => (object) []]],
                'stop_reason' => 'tool_use',
                'usage' => ['input_tokens' => 10, 'output_tokens' => 5],
            ]),
            Http::response([
                'id' => 'msg_2',
                'type' => 'message',
                'role' => 'assistant',
                'model' => 'claude-sonnet-4-6',
                'content' => [['type' => 'text', 'text' => 'The number is 72019.']],
                'stop_reason' => 'end_turn',
                'usage' => ['input_tokens' => 10, 'output_tokens' => 5],
            ]),
        ]),
    ]);

    $user = (object) ['id' => 1];

    $paused = (new RememberingToolSelectorAgent)->forUser($user)->prompt('Generate a number', provider: 'anthropic');

    ApprovableNumberGenerator::$invocations = 0;

    (new RememberingToolSelectorAgent)
        ->continue($paused->conversationId, $user)
        ->stream(Decisions::from(['toolu_1' => true]), provider: 'anthropic')
        ->each(fn (): bool => true);

    expect(ApprovableNumberGenerator::$invocations)->toBe(1)
        ->and($classifications)->toBe(1);
});
