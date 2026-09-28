<?php

namespace Tests\Fixtures\Agents;

use Closure;
use Laravel\Ai\Contracts\Agent;
use Laravel\Ai\Contracts\HasMiddleware;
use Laravel\Ai\Contracts\HasTools;
use Laravel\Ai\PendingStep;
use Laravel\Ai\Promptable;
use Laravel\Ai\Tools\ToolSelector;
use Tests\Fixtures\Tools\FixedNumberGenerator;
use Tests\Fixtures\Tools\NamedTool;

final class ToolSelectorAgent implements Agent, HasMiddleware, HasTools
{
    use Promptable;

    public array $providerOptions = [];

    public array $stepTools = [];

    public function __construct(private ?ToolSelector $selector = null) {}

    public function instructions(): string
    {
        return 'You are a helpful assistant.';
    }

    public function tools(): iterable
    {
        return [
            $this->selector ?? new ToolSelector([
                new NamedTool,
                new FixedNumberGenerator,
            ]),
        ];
    }

    public function middleware(): array
    {
        return [function (PendingStep $step, Closure $next) {
            $this->providerOptions[] = $step->options?->providerOptions($step->provider);
            $this->stepTools[] = $step->tools;

            return $next($step);
        }];
    }
}
