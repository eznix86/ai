<?php

use Laravel\Ai\Ai;
use Laravel\Ai\Classification;
use Laravel\Ai\Enums\Lab;
use Laravel\Ai\Messages\AssistantMessage;
use Laravel\Ai\Messages\Message;
use Laravel\Ai\Messages\UserMessage;
use Laravel\Ai\Prompts\ClassificationPrompt;
use Laravel\Ai\Providers\Provider;
use Laravel\Ai\Responses\Data\ChoiceAnswer;
use Laravel\Ai\Responses\Data\ToolCall;
use Laravel\Ai\Tools\ToolSelectionContext;
use Laravel\Ai\Tools\ToolSelector;
use Tests\Fixtures\Agents\ToolSelectorAgent;
use Tests\Fixtures\Tools\FixedNumberGenerator;
use Tests\Fixtures\Tools\NamedTool;

test('selects one tool by classifying the prompt', function (): void {
    Classification::fake([
        ['tool' => new ChoiceAnswer('custom_named_tool', [
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
        ]);
});

test('classifier provider options do not become agent provider options', function (): void {
    $classificationProvider = null;

    Classification::fake([
        ['tool' => new ChoiceAnswer('custom_named_tool', [
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

        return ['tool' => new ChoiceAnswer('custom_named_tool', [
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
        ['tool' => new ChoiceAnswer('custom_named_tool', [
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
        ['tool' => new ChoiceAnswer('custom_named_tool', [
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
