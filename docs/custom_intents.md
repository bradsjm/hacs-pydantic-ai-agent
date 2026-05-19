Home Assistant uses a multi-stage intent recognition system based on the `hassil` library, which matches user text against predefined sentence templates rather than simple keywords or regex patterns. The system progresses through several matching stages with increasing flexibility.

## Matching Stages

The `DefaultAgent` class implements a four-stage matching process in the `_recognize` method [2](#0-1) :

1. **EXPOSED_ENTITIES_ONLY** - Strict matching against only entities explicitly exposed to Assist [3](#0-2) 
2. **FUZZY** - Uses `FuzzyNgramMatcher` to guess intents when exact match fails [4](#0-3) 
3. **UNEXPOSED_ENTITIES** - Fallback to match against all entities including unexposed ones [5](#0-4) 
4. **UNKNOWN_NAMES** - Captures names not known to Home Assistant for error messages [6](#0-5) 

## Matching Mechanisms

### Strict Matching
Uses `recognize_best()` from hassil to find the best match against sentence templates with exact wording [7](#0-6) . This filters entities by input text before matching to improve performance [8](#0-7) .

### Fuzzy Matching
Uses `FuzzyNgramMatcher` with n-gram models to handle variations in wording [9](#0-8) . This is loaded from the `home-assistant-intents` package which contains pre-built models [10](#0-9) .

### Sentence Templates
Intents are defined as sentence templates in the `home-assistant-intents` package, not keywords or regex [11](#0-10) . Custom sentences can be added via YAML files in `<config>/custom_sentences/<language>/` [12](#0-11) .

## Sentence Triggers
Separate from intents, sentence triggers allow automation-specific matching using the same hassil recognition system [13](#0-12) .

### Citations

**File:** homeassistant/components/assist_pipeline/pipeline.py (L1081-1126)
```python
    async def recognize_intent(
        self,
        intent_input: str,
        conversation_id: str,
        conversation_extra_system_prompt: str | None,
    ) -> tuple[str, bool]:
        """Run intent recognition portion of pipeline.

        Returns (speech, all_targets_in_satellite_area).
        """
        if self.intent_agent is None or self._conversation_data is None:
            raise RuntimeError("Recognize intent was not prepared")

        if self.pipeline.conversation_language == MATCH_ALL:
            # LLMs support all languages ('*') so use languages from the
            # pipeline for intent fallback.
            #
            # We prioritize the STT and TTS languages because they may be more
            # specific, such as "zh-CN" instead of just "zh". This is necessary
            # for languages whose intents are split out by region when
            # preferring local intent matching.
            input_language = (
                self.pipeline.stt_language
                or self.pipeline.tts_language
                or self.pipeline.language
            )
        else:
            input_language = self.pipeline.conversation_language

        self.process_event(
            PipelineEvent(
                PipelineEventType.INTENT_START,
                {
                    "engine": self.intent_agent.id,
                    "language": input_language,
                    "intent_input": intent_input,
                    "conversation_id": conversation_id,
                    "device_id": self._device_id,
                    "satellite_id": self._satellite_id,
                    "prefer_local_intents": self.pipeline.prefer_local_intents,
                },
            )
        )

        try:
            if self.tts_stream and self.tts_stream.supports_streaming_input:
```

**File:** homeassistant/components/conversation/default_agent.py (L37-47)
```python
from home_assistant_intents import (
    ErrorKey,
    FuzzyConfig,
    FuzzyLanguageResponses,
    LanguageScores,
    get_fuzzy_config,
    get_fuzzy_language,
    get_intents,
    get_language_scores,
    get_languages,
)
```

**File:** homeassistant/components/conversation/default_agent.py (L135-136)
```python
    EXPOSED_ENTITIES_ONLY = auto()
    """Match against exposed entities only."""
```

**File:** homeassistant/components/conversation/default_agent.py (L141-142)
```python
    UNEXPOSED_ENTITIES = auto()
    """Match against unexposed entities in Home Assistant."""
```

**File:** homeassistant/components/conversation/default_agent.py (L356-362)
```python
        if self._exposed_names_trie is not None:
            # Filter by input string
            text = remove_punctuation(user_input.text).strip().lower()
            slot_lists["name"] = TextSlotList(
                name="name",
                values=[result[2] for result in self._exposed_names_trie.find(text)],
            )
```

**File:** homeassistant/components/conversation/default_agent.py (L655-817)
```python
    def _recognize(
        self,
        user_input: ConversationInput,
        lang_intents: LanguageIntents,
        slot_lists: dict[str, SlotList],
        intent_context: dict[str, Any] | None,
        language: str,
        strict_intents_only: bool,
    ) -> RecognizeResult | None:
        """Search intents for a match to user input."""
        skip_exposed_match = False

        # Try cache first
        cache_key = IntentCacheKey(
            text=user_input.text,
            language=language,
            satellite_id=user_input.satellite_id,
        )
        cache_value = self._intent_cache.get(cache_key)
        if cache_value is not None:
            if (cache_value.result is not None) and (
                cache_value.stage == IntentMatchingStage.EXPOSED_ENTITIES_ONLY
            ):
                _LOGGER.debug("Got cached result for exposed entities")
                return cache_value.result

            # Continue with matching, but we know we won't succeed for exposed
            # entities only.
            skip_exposed_match = True

        if not skip_exposed_match:
            start_time = time.monotonic()
            strict_result = self._recognize_strict(
                user_input, lang_intents, slot_lists, intent_context, language
            )
            _LOGGER.debug(
                "Checked exposed entities in %s second(s)",
                time.monotonic() - start_time,
            )

            # Update cache
            self._intent_cache.put(
                cache_key,
                IntentCacheValue(
                    result=strict_result,
                    stage=IntentMatchingStage.EXPOSED_ENTITIES_ONLY,
                ),
            )

            if strict_result is not None:
                # Successful strict match with exposed entities
                return strict_result

        if strict_intents_only:
            # Don't try matching against all entities or doing a fuzzy match
            return None

        # Use fuzzy matching
        skip_fuzzy_match = False
        if cache_value is not None:
            if (cache_value.result is not None) and (
                cache_value.stage == IntentMatchingStage.FUZZY
            ):
                _LOGGER.debug("Got cached result for fuzzy match")
                return cache_value.result

            # Continue with matching, but we know we won't succeed for fuzzy
            # match.
            skip_fuzzy_match = True

        if (not skip_fuzzy_match) and self.fuzzy_matching:
            start_time = time.monotonic()
            fuzzy_result = self._recognize_fuzzy(lang_intents, user_input)

            # Update cache
            self._intent_cache.put(
                cache_key,
                IntentCacheValue(result=fuzzy_result, stage=IntentMatchingStage.FUZZY),
            )

            _LOGGER.debug(
                "Did fuzzy match in %s second(s)", time.monotonic() - start_time
            )

            if fuzzy_result is not None:
                return fuzzy_result

        # Try again with all entities (including unexposed)
        skip_unexposed_entities_match = False
        if cache_value is not None:
            if (cache_value.result is not None) and (
                cache_value.stage == IntentMatchingStage.UNEXPOSED_ENTITIES
            ):
                _LOGGER.debug("Got cached result for all entities")
                return cache_value.result

            # Continue with matching, but we know we won't succeed for all
            # entities.
            skip_unexposed_entities_match = True

        if not skip_unexposed_entities_match:
            unexposed_entities_slot_lists = {
                **slot_lists,
                "name": self._get_unexposed_entity_names(user_input.text),
            }

            start_time = time.monotonic()
            strict_result = self._recognize_strict(
                user_input,
                lang_intents,
                unexposed_entities_slot_lists,
                intent_context,
                language,
            )

            _LOGGER.debug(
                "Checked all entities in %s second(s)", time.monotonic() - start_time
            )

            # Update cache
            self._intent_cache.put(
                cache_key,
                IntentCacheValue(
                    result=strict_result, stage=IntentMatchingStage.UNEXPOSED_ENTITIES
                ),
            )

            if strict_result is not None:
                # Not a successful match, but useful for an error message.
                # This should fail the intent handling phase (async_match_targets).
                return strict_result

        # Check unknown names
        skip_unknown_names = False
        if cache_value is not None:
            if (cache_value.result is not None) and (
                cache_value.stage == IntentMatchingStage.UNKNOWN_NAMES
            ):
                _LOGGER.debug("Got cached result for unknown names")
                return cache_value.result

            skip_unknown_names = True

        maybe_result: RecognizeResult | None = None
        if not skip_unknown_names:
            start_time = time.monotonic()
            maybe_result = self._recognize_unknown_names(
                lang_intents, user_input, slot_lists, intent_context
            )

            # Update cache
            self._intent_cache.put(
                cache_key,
                IntentCacheValue(
                    result=maybe_result, stage=IntentMatchingStage.UNKNOWN_NAMES
                ),
            )

            _LOGGER.debug(
                "Did unknown names match in %s second(s)", time.monotonic() - start_time
            )

        return maybe_result
```

**File:** homeassistant/components/conversation/default_agent.py (L819-867)
```python
    def _recognize_fuzzy(
        self, lang_intents: LanguageIntents, user_input: ConversationInput
    ) -> RecognizeResult | None:
        """Return fuzzy recognition from hassil."""
        if lang_intents.fuzzy_matcher is None:
            return None

        context_area: str | None = None
        satellite_area, _ = self._get_satellite_area_and_device(
            user_input.satellite_id, user_input.device_id
        )
        if satellite_area:
            context_area = satellite_area.name

        fuzzy_result = lang_intents.fuzzy_matcher.match(
            user_input.text, context_area=context_area
        )
        if fuzzy_result is None:
            return None

        response = "default"
        if lang_intents.fuzzy_responses:
            domain = ""  # no domain
            if "name" in fuzzy_result.slots:
                domain = fuzzy_result.name_domain
            elif "domain" in fuzzy_result.slots:
                domain = fuzzy_result.slots["domain"].value

            slot_combo = tuple(sorted(fuzzy_result.slots))
            if (
                intent_responses := lang_intents.fuzzy_responses.get(
                    fuzzy_result.intent_name
                )
            ) and (combo_responses := intent_responses.get(slot_combo)):
                response = combo_responses.get(domain, response)

        entities = [
            MatchEntity(name=slot_name, value=slot_value.value, text=slot_value.text)
            for slot_name, slot_value in fuzzy_result.slots.items()
        ]

        return RecognizeResult(
            intent=Intent(name=fuzzy_result.intent_name),
            intent_data=IntentData(sentence_texts=[]),
            intent_metadata={METADATA_FUZZY_MATCH: True},
            entities={entity.name: entity for entity in entities},
            entities_list=entities,
            response=response,
        )
```

**File:** homeassistant/components/conversation/default_agent.py (L869-947)
```python
    def _recognize_unknown_names(
        self,
        lang_intents: LanguageIntents,
        user_input: ConversationInput,
        slot_lists: dict[str, SlotList],
        intent_context: dict[str, Any] | None,
    ) -> RecognizeResult | None:
        """Return result with unknown names for an error message."""
        maybe_result: RecognizeResult | None = None

        best_num_matched_entities = 0
        best_num_unmatched_entities = 0
        best_num_unmatched_ranges = 0
        for result in recognize_all(
            user_input.text,
            lang_intents.intents,
            slot_lists=slot_lists,
            intent_context=intent_context,
            allow_unmatched_entities=True,
        ):
            if result.text_chunks_matched < 1:
                # Skip results that don't match any literal text
                continue

            # Don't count missing entities that couldn't be filled from context
            num_matched_entities = 0
            for matched_entity in result.entities_list:
                if matched_entity.name not in result.unmatched_entities:
                    num_matched_entities += 1

            num_unmatched_entities = 0
            num_unmatched_ranges = 0
            for unmatched_entity in result.unmatched_entities_list:
                if isinstance(unmatched_entity, UnmatchedTextEntity):
                    if unmatched_entity.text != MISSING_ENTITY:
                        num_unmatched_entities += 1
                elif isinstance(unmatched_entity, UnmatchedRangeEntity):
                    num_unmatched_ranges += 1
                    num_unmatched_entities += 1
                else:
                    num_unmatched_entities += 1

            if (
                (maybe_result is None)  # first result
                or (
                    # More literal text matched
                    result.text_chunks_matched > maybe_result.text_chunks_matched
                )
                or (
                    # More entities matched
                    num_matched_entities > best_num_matched_entities
                )
                or (
                    # Fewer unmatched entities
                    (num_matched_entities == best_num_matched_entities)
                    and (num_unmatched_entities < best_num_unmatched_entities)
                )
                or (
                    # Prefer unmatched ranges
                    (num_matched_entities == best_num_matched_entities)
                    and (num_unmatched_entities == best_num_unmatched_entities)
                    and (num_unmatched_ranges > best_num_unmatched_ranges)
                )
                or (
                    # Prefer match failures with entities
                    (result.text_chunks_matched == maybe_result.text_chunks_matched)
                    and (num_unmatched_entities == best_num_unmatched_entities)
                    and (num_unmatched_ranges == best_num_unmatched_ranges)
                    and (
                        ("name" in result.entities)
                        or ("name" in result.unmatched_entities)
                    )
                )
            ):
                maybe_result = result
                best_num_matched_entities = num_matched_entities
                best_num_unmatched_entities = num_unmatched_entities
                best_num_unmatched_ranges = num_unmatched_ranges

```

**File:** homeassistant/components/conversation/default_agent.py (L1001-1018)
```python
    def _recognize_strict(
        self,
        user_input: ConversationInput,
        lang_intents: LanguageIntents,
        slot_lists: dict[str, SlotList],
        intent_context: dict[str, Any] | None,
        language: str,
    ) -> RecognizeResult | None:
        """Search intents for a strict match to user input."""
        return recognize_best(
            user_input.text,
            lang_intents.intents,
            slot_lists=slot_lists,
            intent_context=intent_context,
            language=language,
            best_metadata_key=METADATA_CUSTOM_SENTENCE,
            best_slot_name="name",
        )
```

**File:** homeassistant/components/conversation/default_agent.py (L1154-1165)
```python
        lang_variant_intents = get_intents(language_variant, json_load=json_load)

        if lang_variant_intents:
            # Merge sentences into existing dictionary
            # Overriding because source dict is empty
            intents_dict = lang_variant_intents

            _LOGGER.debug(
                "Loaded built-in intents for language=%s (%s)",
                language,
                language_variant,
            )
```

**File:** homeassistant/components/conversation/default_agent.py (L1167-1208)
```python
        # Check for custom sentences in <config>/custom_sentences/<language>/
        custom_sentences_dir = Path(
            self.hass.config.path("custom_sentences", language_variant)
        )
        if custom_sentences_dir.is_dir():
            for custom_sentences_path in custom_sentences_dir.rglob("*.yaml"):
                with custom_sentences_path.open(
                    encoding="utf-8"
                ) as custom_sentences_file:
                    # Merge custom sentences
                    if not isinstance(
                        custom_sentences_yaml := yaml.safe_load(custom_sentences_file),
                        dict,
                    ):
                        _LOGGER.warning(
                            "Custom sentences file does not match expected format path=%s",
                            custom_sentences_file.name,
                        )
                        continue

                    # Add metadata so we can identify custom sentences in the debugger
                    custom_intents_dict = custom_sentences_yaml.get("intents", {})
                    for intent_dict in custom_intents_dict.values():
                        intent_data_list = intent_dict.get("data", [])
                        for intent_data in intent_data_list:
                            sentence_metadata = intent_data.get("metadata", {})
                            sentence_metadata[METADATA_CUSTOM_SENTENCE] = True
                            sentence_metadata[METADATA_CUSTOM_FILE] = str(
                                custom_sentences_path.relative_to(
                                    custom_sentences_dir.parent
                                )
                            )
                            intent_data["metadata"] = sentence_metadata

                    merge_dict(intents_dict, custom_sentences_yaml)

                _LOGGER.debug(
                    "Loaded custom sentences language=%s (%s), path=%s",
                    language,
                    language_variant,
                    custom_sentences_path,
                )
```

**File:** homeassistant/components/conversation/default_agent.py (L1261-1291)
```python
        fuzzy_matcher = FuzzyNgramMatcher(
            intents=intents,
            intent_models={
                intent_name: Sqlite3NgramModel(
                    order=fuzzy_model.order,
                    words={
                        word: str(word_id)
                        for word, word_id in fuzzy_model.words.items()
                    },
                    database_path=fuzzy_model.database_path,
                )
                for intent_name, fuzzy_model in fuzzy_info.ngram_models.items()
            },
            intent_slot_list_names=self._fuzzy_config.slot_list_names,
            slot_combinations={
                intent_name: {
                    combo_key: SlotCombinationInfo(
                        context_area=combo_info.context_area,
                        name_domains=(
                            set(combo_info.name_domains)
                            if combo_info.name_domains
                            else None
                        ),
                    )
                    for combo_key, combo_info in intent_combos.items()
                }
                for intent_name, intent_combos in self._fuzzy_config.slot_combinations.items()
            },
            domain_keywords=fuzzy_info.domain_keywords,
            stop_words=fuzzy_info.stop_words,
        )
```

**File:** homeassistant/components/conversation/default_agent.py (L1502-1546)
```python
    async def async_recognize_sentence_trigger(
        self, user_input: ConversationInput
    ) -> SentenceTriggerResult | None:
        """Try to match sentence against registered trigger sentences.

        Calls the registered callbacks if there's a match and returns a sentence
        trigger result.
        """
        if not self._trigger_intents_config.get("intents"):
            # No triggers registered
            return None

        if self._trigger_intents is None:
            # Need to rebuild intents before matching
            self._rebuild_trigger_intents()

        assert self._trigger_intents is not None

        matched_triggers: dict[str, RecognizeResult] = {}
        matched_template: str | None = None
        for result in recognize_all(user_input.text, self._trigger_intents):
            if result.intent_sentence is not None:
                matched_template = result.intent_sentence.text

            trigger_intent_name = result.intent.name
            if trigger_intent_name in matched_triggers:
                # Already matched a sentence from this trigger
                break

            matched_triggers[trigger_intent_name] = result

        if not matched_triggers:
            # Sentence did not match any trigger sentences
            return None

        _LOGGER.debug(
            "'%s' matched %s trigger(s): %s",
            user_input.text,
            len(matched_triggers),
            list(matched_triggers),
        )

        return SentenceTriggerResult(
            user_input.text, matched_template, matched_triggers
        )
```

Here is a complete, self-contained example based on the patterns used in `assist_satellite/entity.py` and `default_agent.py`.

## How it works

The key insight is that `Intents.from_dict()` accepts the same dict structure as the YAML files, so you can build it programmatically at runtime. The `_rebuild_trigger_intents` method in `DefaultAgent` is the canonical example of this pattern. [1](#4-0) 

The `assist_satellite` entity does the same thing for question-answer matching: [2](#4-1) [3](#4-2) 

## Complete example

```python
# custom_components/my_llm_agent/conversation.py

from __future__ import annotations

from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from typing import Any

from hassil import Intents, recognize
from hassil.expression import Expression, Group, ListReference
from hassil.intents import WildcardSlotList
from hassil.recognize import RecognizeResult, recognize_best

from homeassistant.components import conversation
from homeassistant.helpers import intent


# Type alias for a callback that receives matched slots and returns a response string
IntentCallback = Callable[[dict[str, Any]], Coroutine[Any, Any, str]]


@dataclass
class _RegisteredIntent:
    sentences: list[str]
    callback: IntentCallback


class MyLLMConversationEntity(
    conversation.ConversationEntity,
    conversation.AbstractConversationAgent,
):
    """Custom LLM conversation agent with dynamic intent bypass."""

    def __init__(self, ...) -> None:
        super().__init__(...)
        # Map of intent_name -> registered intent
        self._custom_intents: dict[str, _RegisteredIntent] = {}
        # Cached compiled Intents object; None means needs rebuild
        self._compiled_intents: Intents | None = None

    # ------------------------------------------------------------------
    # Public API for dynamic registration
    # ------------------------------------------------------------------

    def register_intent(
        self,
        intent_name: str,
        sentences: list[str],
        callback: IntentCallback,
    ) -> Callable[[], None]:
        """Register sentences for an intent. Returns an unregister function.

        Sentence template syntax (hassil):
          - Optional words:   [optional word]
          - Alternatives:     (this|that)
          - Wildcard slot:    {slot_name}   (captured as a string)

        Example sentences:
          ["play {song} [please]", "put on (the song|) {song}"]
        """
        self._custom_intents[intent_name] = _RegisteredIntent(
            sentences=sentences,
            callback=callback,
        )
        self._compiled_intents = None  # invalidate cache

        def unregister() -> None:
            self._custom_intents.pop(intent_name, None)
            self._compiled_intents = None

        return unregister

    # ------------------------------------------------------------------
    # Internal: build/cache the Intents object
    # ------------------------------------------------------------------

    def _get_compiled_intents(self) -> Intents | None:
        """Return (and lazily build) the compiled Intents object."""
        if not self._custom_intents:
            return None

        if self._compiled_intents is not None:
            return self._compiled_intents

        intents_dict = {
            "language": self.hass.config.language,
            "intents": {
                intent_name: {
                    "data": [{"sentences": reg.sentences}]
                }
                for intent_name, reg in self._custom_intents.items()
            },
        }

        compiled = Intents.from_dict(intents_dict)

        # Treat any {slot_name} references as wildcards (captures any text)
        wildcard_names: set[str] = set()
        for intent_obj in compiled.intents.values():
            for intent_data in intent_obj.data:
                for sentence in intent_data.sentences:
                    _collect_list_references(sentence.expression, wildcard_names)

        for wildcard_name in wildcard_names:
            compiled.slot_lists[wildcard_name] = WildcardSlotList(wildcard_name)

        self._compiled_intents = compiled
        return compiled

    def _match_custom_intent(self, text: str) -> RecognizeResult | None:
        """Try to match text against registered intents (runs in executor)."""
        compiled = self._get_compiled_intents()
        if compiled is None:
            return None
        return recognize_best(text, compiled)

    # ------------------------------------------------------------------
    # Override _async_handle_message to check intents first
    # ------------------------------------------------------------------

    async def _async_handle_message(
        self,
        user_input: conversation.ConversationInput,
        chat_log: conversation.ChatLog,
    ) -> conversation.ConversationResult:
        """Handle a message: check custom intents before calling the LLM."""

        # hassil matching is CPU-bound; run in executor like DefaultAgent does
        result: RecognizeResult | None = await self.hass.async_add_executor_job(
            self._match_custom_intent, user_input.text
        )

        if result is not None and not result.unmatched_entities:
            # Extract slot values from the match
            slots = {
                entity.name: entity.value
                for entity in result.entities_list
            }
            intent_name = result.intent.name
            registered = self._custom_intents.get(intent_name)

            if registered is not None:
                response_text = await registered.callback(slots)

                # Build a ConversationResult without touching the LLM
                intent_response = intent.IntentResponse(
                    language=user_input.language
                )
                intent_response.async_set_speech(response_text)
                return conversation.ConversationResult(
                    response=intent_response,
                    conversation_id=chat_log.conversation_id,
                )

        # No custom intent matched — fall through to your LLM logic
        return await self._call_llm(user_input, chat_log)

    async def _call_llm(self, user_input, chat_log):
        """Your existing LLM call goes here."""
        ...


def _collect_list_references(expression: Expression, list_names: set[str]) -> None:
    """Recursively collect {slot_name} references from a hassil expression."""
    if isinstance(expression, Group):
        for item in expression.items:
            _collect_list_references(item, list_names)
    elif isinstance(expression, ListReference):
        list_names.add(expression.slot_name)
```

This helper is copied verbatim from `default_agent.py`: [4](#4-3) 

## Usage in `async_setup_entry`

```python
async def async_setup_entry(hass, entry):
    entity = MyLLMConversationEntity(...)

    async def handle_play_song(slots: dict) -> str:
        song = slots.get("song", "something")
        # do whatever you need...
        return f"Playing {song}"

    unregister = entity.register_intent(
        intent_name="PlaySong",
        sentences=["play {song} [please]", "put on {song}"],
        callback=handle_play_song,
    )

    # Clean up when the entry is unloaded
    entry.async_on_unload(unregister)
```

## Key points

- `Intents.from_dict()` accepts the same structure as the YAML files, so the sentence template syntax is identical (optional words `[...]`, alternatives `(a|b)`, slots `{name}`).
- Any `{slot_name}` in a template is automatically treated as a wildcard (captures any text) by the `_collect_list_references` + `WildcardSlotList` step. If you want a fixed list of values for a slot instead, add it to `compiled.slot_lists` as a `TextSlotList`.
- `recognize_best` returns `None` on no match, so the fallthrough to the LLM is clean.
- The matching runs in `async_add_executor_job` because hassil is synchronous and CPU-bound, consistent with how `DefaultAgent` handles it. [5](#4-4)

### Citations

**File:** homeassistant/components/conversation/default_agent.py (L366-374)
```python
        result = await self.hass.async_add_executor_job(
            self._recognize,
            user_input,
            lang_intents,
            slot_lists,
            intent_context,
            language,
            strict_intents_only,
        )
```

**File:** homeassistant/components/conversation/default_agent.py (L1480-1498)
```python
    def _rebuild_trigger_intents(self) -> None:
        """Rebuild the HassIL intents object from the trigger intents dict."""
        intents_dict = {
            "language": self.hass.config.language,
            **self._trigger_intents_config,
        }
        trigger_intents = Intents.from_dict(intents_dict)

        # Assume slot list references are wildcards
        wildcard_names: set[str] = set()
        for trigger_intent in trigger_intents.intents.values():
            for intent_data in trigger_intent.data:
                for sentence in intent_data.sentences:
                    _collect_list_references(sentence.expression, wildcard_names)

        for wildcard_name in wildcard_names:
            trigger_intents.slot_lists[wildcard_name] = WildcardSlotList(wildcard_name)

        self._trigger_intents = trigger_intents
```

**File:** homeassistant/components/conversation/default_agent.py (L1858-1865)
```python
def _collect_list_references(expression: Expression, list_names: set[str]) -> None:
    """Collect list reference names recursively."""
    if isinstance(expression, Group):
        for item in expression.items:
            _collect_list_references(item, list_names)
    elif isinstance(expression, ListReference):
        # {list}
        list_names.add(expression.slot_name)
```

**File:** homeassistant/components/assist_satellite/entity.py (L13-15)
```python
from hassil import Intents, recognize
from hassil.expression import Expression, Group, ListReference
from hassil.intents import WildcardSlotList
```

**File:** homeassistant/components/assist_satellite/entity.py (L394-422)
```python
        intents = Intents.from_dict(
            {
                "language": self.hass.config.language,
                "intents": {
                    "QuestionIntent": {
                        "data": [
                            {
                                "sentences": answer["sentences"],
                                "metadata": {"answer_id": answer["id"]},
                            }
                            for answer in answers
                        ]
                    }
                },
            }
        )

        # Assume slot list references are wildcards
        wildcard_names: set[str] = set()
        for intent in intents.intents.values():
            for intent_data in intent.data:
                for sentence in intent_data.sentences:
                    _collect_list_references(sentence.expression, wildcard_names)

        for wildcard_name in wildcard_names:
            intents.slot_lists[wildcard_name] = WildcardSlotList(wildcard_name)

        # Match response text
        result = recognize(response_text, intents)
```
