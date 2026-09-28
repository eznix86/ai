<?php

use Illuminate\Contracts\JsonSchema\JsonSchema;
use Laravel\Ai\Ai;
use Laravel\Ai\Contracts\Tool;
use Laravel\Ai\Messages\AssistantMessage;
use Laravel\Ai\Messages\UserMessage;
use Laravel\Ai\Responses\Data\ToolCall;
use Laravel\Ai\Streaming\Events\ToolResult;
use Laravel\Ai\Tools\Request;
use Laravel\Ai\Tools\ToolNameResolver;
use Laravel\Ai\Tools\ToolSelectionContext;
use Laravel\Ai\Tools\ToolSelector;
use Tests\Fixtures\Agents\ToolSelectorAgent;
use Tests\Fixtures\Tools\FixedNumberGenerator;
use Tests\Fixtures\Tools\NamedTool;

class SupportTool implements Tool
{
    public function __construct(private string $toolName, private string $toolDescription) {}

    public function name(): string
    {
        return $this->toolName;
    }

    public function description(): string
    {
        return $this->toolDescription;
    }

    public function handle(Request $request): string
    {
        return "ran {$this->toolName}";
    }

    public function schema(JsonSchema $schema): array
    {
        return [];
    }
}

function supportSelector(string $provider): ToolSelector
{
    return (new ToolSelector([
        new SupportTool('issue_refund', 'Refund a payment to the customer: double charges, returned items, money back.'),
        new SupportTool('track_order', 'Look up where a shipped order is and when the package will arrive.'),
        new SupportTool('cancel_subscription', 'Stop a recurring subscription plan so the customer is no longer billed.'),
        new SupportTool('update_address', 'Change the shipping or billing address on the customer account.'),
        new SupportTool('reset_password', 'Help a customer who cannot log in by sending a password reset link.'),
    ], 'Which support action handles the customer request?'))->using($provider);
}

test('selects from the current prompt', function (string $provider, string $apiKey): void {
    requiresApiKey($apiKey);

    $tool = (new ToolSelector([new NamedTool, new FixedNumberGenerator]))
        ->using($provider)
        ->select('Generate a cryptographically secure random number.');

    expect($tool)->toBeInstanceOf(FixedNumberGenerator::class);
})->with('classification-providers');

test('selects from the most recent conversation turn', function (string $provider, string $apiKey): void {
    requiresApiKey($apiKey);

    $selector = supportSelector($provider)->withConversationContext();

    $tool = $selector->select(new ToolSelectionContext(
        new ToolSelectorAgent($selector),
        'Yes, go ahead and do it.',
        [
            new UserMessage('Where is order 100?'),
            new AssistantMessage('It was delivered on Monday.'),
            new UserMessage('Thanks. Now I want to cancel my premium membership.'),
            new AssistantMessage('You will lose access at the end of the month. Should I continue?'),
        ],
    ));

    expect(ToolNameResolver::resolve($tool))->toBe('cancel_subscription');
})->with('classification-providers');

test('selects from custom state', function (string $provider, string $apiKey): void {
    requiresApiKey($apiKey);

    $selector = supportSelector($provider)->withState(fn (ToolSelectionContext $context): array => [
        'customer_request' => $context->prompt,
        'last_order_status' => 'in transit, carrier DHL',
    ]);

    $tool = $selector->select(new ToolSelectionContext(new ToolSelectorAgent($selector), 'Any news on my order?', []));

    expect(ToolNameResolver::resolve($tool))->toBe('track_order');
})->with('classification-providers');

test('selects a tool only when the request asks for one', function (string $provider, string $apiKey, string $prompt, array $acceptable): void {
    requiresApiKey($apiKey);

    $tool = supportSelector($provider)->select($prompt);

    expect($tool === null ? null : ToolNameResolver::resolve($tool))->toBeIn($acceptable);
})->with('classification-providers')->with([
    'off topic: weather' => ['What is the weather in Geneva tomorrow?', [null]],
    'off topic: joke' => ['Tell me a joke about cats.', [null]],
    'off topic: opening hours' => ['What are your store opening hours?', [null]],
    'off topic: gift cards' => ['Do you sell gift cards?', [null]],
    'off topic: sofa color' => ['Can I change the color of the sofa I ordered?', [null]],
    'refund' => ['I was charged twice for order #4821, I want my money back.', ['issue_refund']],
    'tracking' => ['Where is my package? It was supposed to arrive yesterday.', ['track_order']],
    'cancellation' => ['Please stop my monthly plan, I do not want to pay anymore.', ['cancel_subscription']],
    'address' => ['I moved. Send future orders to 12 Rue du Lac, Geneva.', ['update_address']],
    'password' => ['I forgot my password and cannot sign in.', ['reset_password']],
    'refund after return' => ['The blender arrived broken, I returned it last week, where is my refund?', ['issue_refund']],
    'stale tracking' => ['My tracking number shows no update for five days.', ['track_order']],
    'locked out' => ['I am locked out of my account, the login keeps failing.', ['reset_password']],
    'partial: pause subscription' => ['Can I pause my subscription for two months instead of cancelling?', [null]],
    'partial: return before refund' => ['I want to return these shoes, how do I send them back?', [null]],
    'partial: missing order refund' => ['My order never arrived and I want my money back.', ['issue_refund', 'track_order']],
    'partial: email change' => ['Change the email address on my account.', [null]],
    'partial: card change' => ['Update the credit card used for my subscription.', [null]],
    'partial: discount' => ['Can I get a discount on my next order?', [null]],
    'partial: account breach' => ['I think someone else logged into my account.', ['reset_password', null]],
    'partial: refund timing' => ['How long does a refund take to reach my bank?', [null]],
    'partial: subscription status' => ['Is my subscription still active?', [null]],
    'partial: upgrade' => ['Upgrade my plan to premium.', [null]],
    'partial: delivered not received' => ['The package shows delivered but I never got it.', ['track_order', 'issue_refund']],
    'partial: delivery instructions' => ['Leave my deliveries at the back door from now on.', ['update_address', null]],
]);

test('selects the tool for a full agent run', function (string $provider, string $apiKey): void {
    requiresApiKey($apiKey);

    $agent = new ToolSelectorAgent(supportSelector($provider));

    Ai::fakeAgent($agent::class, [
        new ToolCall('call_1', 'issue_refund', []),
        'Refund issued.',
    ]);

    $response = $agent->prompt('You billed my card twice this month, refund the extra charge.');

    expect($response->toolResults->first()->result)->toBe('ran issue_refund');
})->with('classification-providers');

test('exposes no tool to an agent run when none applies', function (string $provider, string $apiKey): void {
    requiresApiKey($apiKey);

    $agent = new ToolSelectorAgent(supportSelector($provider));

    Ai::fakeAgent($agent::class, ['We are open from 9 to 18.']);

    $agent->prompt('What are your store opening hours?');

    expect($agent->stepTools)->toBe([[]]);
})->with('classification-providers');

test('selects the tool for a streamed agent run', function (string $provider, string $apiKey): void {
    requiresApiKey($apiKey);

    $agent = new ToolSelectorAgent(supportSelector($provider));

    Ai::fakeAgent($agent::class, [
        new ToolCall('call_1', 'reset_password', []),
        'Reset link sent.',
    ]);

    $stream = $agent->stream('I am locked out of my account, the login keeps failing.');
    $stream->each(fn (): bool => true);

    expect($stream->events->first(fn ($event): bool => $event instanceof ToolResult)->toolResult->result)->toBe('ran reset_password');
})->with('classification-providers');
