<?php

namespace Laravel\Ai\Tools;

use Closure;
use InvalidArgumentException;
use Laravel\Ai\Classification;
use Laravel\Ai\Classification\Choice;
use Laravel\Ai\Contracts\Tool;
use Laravel\Ai\Enums\Lab;
use Laravel\Ai\Providers\Provider;
use Laravel\Ai\Responses\Data\ChoiceAnswer;
use Laravel\SerializableClosure\SerializableClosure;
use LogicException;

final class ToolSelector
{
    protected Lab|string|null $provider = null;

    protected ?string $model = null;

    protected int $timeout = 30;

    /** @var array<string, mixed>|SerializableClosure */
    protected array|SerializableClosure $providerOptions = [];

    protected ?int $conversationTurns = null;

    protected ?SerializableClosure $stateResolver = null;

    /**
     * @param  array<Tool>  $tools
     * @param  string|array<string, mixed>  $instructions
     */
    public function __construct(
        public readonly array $tools,
        public readonly string|array $instructions = 'Choose the tool best suited to the current state.',
    ) {
        if (count($tools) < 2) {
            throw new InvalidArgumentException('A tool selector requires at least two tools.');
        }

        foreach ($tools as $tool) {
            if (! $tool instanceof Tool) {
                throw new InvalidArgumentException('Tool selectors may only contain tools.');
            }
        }

        if (count($this->options()) !== count($tools)) {
            throw new InvalidArgumentException('Tool selector names must be unique.');
        }
    }

    public function using(Lab|string $provider, ?string $model = null): self
    {
        $this->provider = $provider;
        $this->model = $model;

        return $this;
    }

    public function timeout(int $seconds = 30): self
    {
        $this->timeout = $seconds;

        return $this;
    }

    /** @param  array<string, mixed>|Closure(Provider): ?array<string, mixed>  $options */
    public function withProviderOptions(array|Closure $options): self
    {
        $this->providerOptions = $options instanceof Closure
            ? new SerializableClosure($options)
            : $options;

        return $this;
    }

    public function withConversationContext(int $turns = 1): self
    {
        if ($turns < 1 || $turns > 5) {
            throw new InvalidArgumentException('Conversation context must contain between one and five turns.');
        }

        $this->conversationTurns = $turns;
        $this->stateResolver = null;

        return $this;
    }

    /** @param  Closure(ToolSelectionContext): (string|array<string, mixed>)  $resolver */
    public function withState(Closure $resolver): self
    {
        $this->stateResolver = new SerializableClosure($resolver);
        $this->conversationTurns = null;

        return $this;
    }

    public function select(string|array|ToolSelectionContext $state): Tool
    {
        $state = $this->resolveState($state);

        $classification = Classification::of($state)
            ->question('tool', new Choice($this->instructions, $this->options()))
            ->timeout($this->timeout);

        $classification->withProviderOptions(
            $this->providerOptions instanceof SerializableClosure
                ? fn (Provider $provider): ?array => ($this->providerOptions)($provider)
                : $this->providerOptions
        );

        $answer = $classification
            ->classify($this->provider, $this->model)
            ->answer('tool');

        if (! $answer instanceof ChoiceAnswer) {
            throw new LogicException('Tool selection requires a choice answer.');
        }

        foreach ($this->tools as $tool) {
            if (ToolNameResolver::resolve($tool) === $answer->choice) {
                return $tool;
            }
        }

        throw new InvalidArgumentException("Classification selected an unknown tool [{$answer->choice}].");
    }

    protected function resolveState(string|array|ToolSelectionContext $state): string|array
    {
        if (! $state instanceof ToolSelectionContext) {
            return $state;
        }

        if ($this->stateResolver instanceof SerializableClosure) {
            return ($this->stateResolver)($state);
        }

        if ($this->conversationTurns !== null) {
            return $state->stateWithConversation($this->conversationTurns);
        }

        return $state->prompt;
    }

    /**
     * @return array<string, string>
     */
    protected function options(): array
    {
        $options = [];

        foreach ($this->tools as $tool) {
            $options[ToolNameResolver::resolve($tool)] = (string) $tool->description();
        }

        return $options;
    }
}
