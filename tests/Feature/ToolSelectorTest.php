<?php

use Laravel\Ai\Ai;
use Laravel\Ai\Classification;
use Laravel\Ai\Enums\Lab;
use Laravel\Ai\Prompts\ClassificationPrompt;
use Laravel\Ai\Providers\Provider;
use Laravel\Ai\Responses\Data\ChoiceAnswer;
use Laravel\Ai\Responses\Data\ToolCall;
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
