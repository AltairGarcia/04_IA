"""
Advanced Prompt Engineering Module

This module provides tools for constructing, managing, and A/B testing
sophisticated prompts for AI models.
"""
import logging
import random
from typing import Dict, Any, Optional, List, Union, Callable
from string import Formatter

logger = logging.getLogger(__name__)

class PromptTemplateError(ValueError):
    """Custom exception for errors related to prompt templates."""
    pass

class PromptTemplate:
    """
    Represents a prompt template with support for dynamic formatting,
    conditional logic, and context integration.
    """
    def __init__(self, template_id: str, content: str, description: Optional[str] = None, version: str = "1.0"):
        self.template_id = template_id
        self.content = content
        self.description = description
        self.version = version
        self._validate_template()

    def _validate_template(self):
        """
        Validates the template content for basic formatting correctness.
        More advanced validation (e.g., for custom conditional syntax) can be added here.
        """
        try:
            # Basic check for Python string formatting compatibility
            Formatter().parse(self.content)
        except ValueError as e:
            raise PromptTemplateError(f"Invalid template format for '{self.template_id}': {e}")

    def _apply_conditional_logic(self, current_content: str, context: Dict[str, Any]) -> str:
        """
        Applies conditional logic to the prompt content.
        This is a placeholder for more sophisticated conditional logic.
        Example: Replace "{{IF condition}}text{{ENDIF}}" blocks.
        For simplicity, this example doesn't implement full conditional logic.
        A more robust solution might use a mini-language or a library.
        """
        # This is a very basic example. A real implementation would need a parser.
        # For instance, one could look for patterns like:
        # {{IF context_var_exists}} Show this text {{ENDIF}}
        # {{IF context_var == 'some_value'}} Show that text {{ENDIF}}
        # This part is highly dependent on the chosen syntax for conditional logic.
        # For now, we'll just return the content as is.
        processed_content = current_content # Placeholder
        
        # Example of a simple conditional replacement (can be expanded)
        # This is illustrative and not a full-fledged conditional engine.
        for key, value in context.items():
            if isinstance(value, bool): # Simple boolean conditional
                # Construct search strings without problematic f-string nesting for literals
                search_if_pattern = "{{IF " + key + "}}"
                replace_if_with = "" if value else "{{IGNORE_BLOCK_START}}" # Placeholder for block removal
                processed_content = processed_content.replace(search_if_pattern, replace_if_with)
                
                search_endif_pattern = "{{ENDIF " + key + "}}"
                replace_endif_with = "" if value else "{{IGNORE_BLOCK_END}}" # Placeholder for block removal
                processed_content = processed_content.replace(search_endif_pattern, replace_endif_with)

        # A more complex system might involve parsing blocks and evaluating conditions.
        # This is left as an extension point.
        return processed_content


    def format(self, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Formats the prompt template with the given context.
        
        Args:
            context: A dictionary of values to substitute into the template.
        
        Returns:
            The formatted prompt string.
            
        Raises:
            PromptTemplateError: If a required context variable is missing.
        """
        if context is None:
            context = {}
        
        try:
            # Step 1: Apply any custom conditional logic (placeholder for now)
            content_with_conditions = self._apply_conditional_logic(self.content, context)

            # Step 2: Perform standard string formatting
            # We need to identify required keys for robust error handling if they are missing.
            required_keys = {fn for _, fn, _, _ in Formatter().parse(content_with_conditions) if fn is not None}
            
            missing_keys = required_keys - context.keys()
            if missing_keys:
                # Allow partial formatting if some keys are for conditional blocks that were removed
                # This requires more sophisticated parsing of conditional blocks.
                # For now, strict checking:
                # raise PromptTemplateError(f"Missing context variables for template '{self.template_id}': {missing_keys}")
                pass # Allow formatting with missing keys, they might be optional or handled by conditionals

            formatted_content = content_with_conditions.format_map(DefaultSafeDict(context))
            return formatted_content
        except KeyError as e:
            raise PromptTemplateError(f"Missing context variable '{e}' for template '{self.template_id}'. Required: {required_keys}")
        except Exception as e:
            logger.error(f"Error formatting template '{self.template_id}': {e}", exc_info=True)
            raise PromptTemplateError(f"Failed to format template '{self.template_id}': {e}")

class DefaultSafeDict(dict):
    """A dictionary that returns a placeholder for missing keys during string formatting."""
    def __missing__(self, key):
        return f"{{{{{key}}}}}" # Returns the key itself in {{key}} format if missing


class PromptManager:
    """
    Manages a collection of prompt templates and facilitates A/B testing.
    """
    def __init__(self):
        self._templates: Dict[str, PromptTemplate] = {}
        self._ab_tests: Dict[str, Dict[str, Any]] = {} # Basic A/B test setup

    def add_template(self, template: PromptTemplate):
        """Adds a new prompt template to the manager."""
        if template.template_id in self._templates:
            logger.warning(f"Template ID '{template.template_id}' already exists. Overwriting.")
        self._templates[template.template_id] = template
        logger.info(f"Added prompt template: {template.template_id} (Version: {template.version})")

    def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """Retrieves a prompt template by its ID."""
        return self._templates.get(template_id)

    def format_prompt(self, template_id: str, context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Formats a prompt using a registered template and context.
        """
        template = self.get_template(template_id)
        if not template:
            logger.error(f"Prompt template '{template_id}' not found.")
            return None
        try:
            return template.format(context)
        except PromptTemplateError as e:
            logger.error(f"Error formatting prompt for template '{template_id}': {e}")
            return None

    # --- A/B Testing ---
    # The A/B testing mechanism here is conceptual and would need to be
    # integrated with analytics and a more robust selection strategy.

    def setup_ab_test(self, test_name: str, variants: List[str], distribution: Optional[List[float]] = None):
        """
        Sets up an A/B test for prompt templates.
        
        Args:
            test_name: A unique name for the A/B test.
            variants: A list of template_ids to be used as variants.
            distribution: Optional list of weights for each variant. If None, equal distribution.
        """
        if not all(variant_id in self._templates for variant_id in variants):
            missing = [v_id for v_id in variants if v_id not in self._templates]
            raise ValueError(f"Cannot setup A/B test '{test_name}'. Missing template IDs: {missing}")
        
        if distribution and len(distribution) != len(variants):
            raise ValueError("Distribution list length must match variants list length.")

        if distribution and not abs(sum(distribution) - 1.0) < 1e-9: # Check if sum is close to 1
            raise ValueError("Distribution weights must sum to 1.0.")

        self._ab_tests[test_name] = {
            "variants": variants,
            "distribution": distribution or ([1.0/len(variants)] * len(variants))
        }
        logger.info(f"A/B test '{test_name}' setup with variants: {variants} and distribution: {self._ab_tests[test_name]['distribution']}")

    def get_variant_for_ab_test(self, test_name: str, user_id: Optional[str] = None) -> Optional[str]:
        """
        Selects a template_id variant for an A/B test.
        
        Args:
            test_name: The name of the A/B test.
            user_id: Optional user identifier for consistent variant assignment (not fully implemented here).
                     A more robust solution would use hashing for consistent assignment.
                     
        Returns:
            A template_id for the selected variant, or None if the test is not found.
        """
        test_setup = self._ab_tests.get(test_name)
        if not test_setup:
            logger.error(f"A/B test '{test_name}' not found.")
            return None
        
        # Simple random choice based on distribution.
        # For consistent user experience, one might hash user_id to pick a variant.
        selected_variant = random.choices(test_setup["variants"], weights=test_setup["distribution"], k=1)[0]
        logger.debug(f"Selected variant '{selected_variant}' for A/B test '{test_name}'.")
        return selected_variant

    def format_prompt_ab_test(self, test_name: str, context: Optional[Dict[str, Any]] = None, user_id: Optional[str] = None) -> Optional[str]:
        """
        Formats a prompt by selecting a variant from an A/B test.
        """
        variant_template_id = self.get_variant_for_ab_test(test_name, user_id)
        if not variant_template_id:
            return None # Error already logged by get_variant_for_ab_test
        
        return self.format_prompt(variant_template_id, context)


# Example Usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    manager = PromptManager()

    # 1. Define and add templates
    template_greet = PromptTemplate(
        template_id="user_greeting",
        content="Hello {name}! Welcome to {platform}. {{IF is_premium}}You have premium access.{{ENDIF is_premium}}",
        description="A simple user greeting."
    )
    template_query = PromptTemplate(
        template_id="product_query",
        content="User {user_id} asked about {product_name}. Details: {query_details}. Context: {optional_context}",
        description="Formats a user query about a product."
    )
    manager.add_template(template_greet)
    manager.add_template(template_query)

    # 2. Format a simple prompt
    greeting_context_1 = {"name": "Alice", "platform": "Our Awesome App", "is_premium": True}
    formatted_greeting_1 = manager.format_prompt("user_greeting", greeting_context_1)
    print(f"Formatted Greeting 1: {formatted_greeting_1}")
    
    greeting_context_2 = {"name": "Bob", "platform": "Our Awesome App", "is_premium": False} # Test conditional part
    formatted_greeting_2 = manager.format_prompt("user_greeting", greeting_context_2)
    print(f"Formatted Greeting 2: {formatted_greeting_2}")


    # 3. Format a prompt with potentially missing optional context
    query_context = {
        "user_id": "user123", 
        "product_name": "SuperWidget", 
        "query_details": "Does it come in blue?"
        # "optional_context" is missing, DefaultSafeDict will handle it
    }
    formatted_query = manager.format_prompt("product_query", query_context)
    print(f"Formatted Query: {formatted_query}")

    # 4. Setup and use A/B testing (conceptual)
    template_greet_v2 = PromptTemplate(
        template_id="user_greeting_v2",
        content="Hey {name}! So glad you're here at {platform}. {{IF is_premium}}Enjoy your premium features!{{ENDIF is_premium}}",
        description="A more casual user greeting."
    )
    manager.add_template(template_greet_v2)
    
    try:
        manager.setup_ab_test("greeting_style_test", ["user_greeting", "user_greeting_v2"])
        
        print("\nA/B Test Results (run multiple times to see variation):")
        for i in range(5):
            ab_test_greeting = manager.format_prompt_ab_test("greeting_style_test", greeting_context_1, user_id=f"user_{i}")
            print(f"  Run {i+1}: {ab_test_greeting}")
            
    except ValueError as e:
        print(f"Error setting up A/B test: {e}")

    # Test formatting non-existent template
    non_existent = manager.format_prompt("does_not_exist", {})
    if non_existent is None:
        print("\nSuccessfully handled formatting of non-existent template.")
        
    # Test template with formatting error (e.g. unclosed brace)
    try:
        error_template = PromptTemplate("error_template", "Hello {name") # Missing closing brace
    except PromptTemplateError as e:
        print(f"\nSuccessfully caught error in template definition: {e}")

    # Test formatting with missing required variable (if strict checking was enabled in PromptTemplate.format)
    # try:
    #     manager.format_prompt("user_greeting", {"platform": "TestPlatform"}) # Missing 'name'
    # except PromptTemplateError as e:
    #     print(f"\nSuccessfully caught missing context variable: {e}")