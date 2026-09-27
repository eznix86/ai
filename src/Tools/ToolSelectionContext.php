<?php

namespace Laravel\Ai\Tools;

use Laravel\Ai\Contracts\Agent;
use Laravel\Ai\Messages\Message;
use Laravel\Ai\Messages\MessageRole;

final readonly class ToolSelectionContext
{
    /** @param  list<Message>  $messages */
    public function __construct(
        public Agent $agent,
        public string $prompt,
        public array $messages,
    ) {}

    public function stateWithConversation(int $turns): array
    {
        $messages = array_values(array_filter(
            $this->messages,
            fn (Message $message): bool => in_array($message->role, [MessageRole::User, MessageRole::Assistant], true)
                && filled($message->content),
        ));

        $userIndexes = array_keys(array_filter(
            $messages,
            fn (Message $message): bool => $message->role === MessageRole::User,
        ));

        $firstIndex = $userIndexes[max(0, count($userIndexes) - $turns)] ?? count($messages);

        return [
            'current_prompt' => $this->prompt,
            'conversation' => array_map(fn (Message $message): array => [
                'role' => $message->role->value,
                'content' => $message->content,
            ], array_slice($messages, $firstIndex)),
        ];
    }
}
