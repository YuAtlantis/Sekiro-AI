import time
from keys.input_keys import perform_action


class ToolManager:
    def __init__(self):
        self.tools = [
            {'name': 'Tool 1', 'usage_cost': 1, 'cooldown': 8, 'last_used': 0},
            {'name': 'Tool 2', 'usage_cost': 2, 'cooldown': 10, 'last_used': 0},
            {'name': 'Tool 3', 'usage_cost': 3, 'cooldown': 12, 'last_used': 0}
        ]
        self.current_tool_index = 0
        self.remaining_uses = 19
        self.tools_exhausted = False

    def change_tool(self):
        perform_action("Z", 0.1)
        self.current_tool_index = (self.current_tool_index + 1) % len(self.tools)

    def use_specific_tool(self, target_tool_index):
        while self.current_tool_index != target_tool_index:
            self.change_tool()
        self.use_tool()

    def use_tool(self):
        current_tool = self.tools[self.current_tool_index]
        current_time = time.time()

        # Check if the tool is on cooldown
        time_since_last_use = current_time - current_tool['last_used']
        if time_since_last_use < current_tool['cooldown']:
            remaining_cooldown = current_tool['cooldown'] - time_since_last_use
            print(f"{current_tool['name']} is on cooldown. Please wait {remaining_cooldown:.2f} seconds.")
            return remaining_cooldown  # Return remaining cooldown time

        if self.remaining_uses > 0:
            # Check if there are enough remaining uses for the tool's usage cost
            if self.remaining_uses >= current_tool['usage_cost']:
                perform_action("3", 0.2)
                self.remaining_uses -= current_tool['usage_cost']
                current_tool['last_used'] = current_time
                print(f"Used {current_tool['name']}. Remaining uses: {self.remaining_uses}")
            else:
                print(f"Not enough uses left for {current_tool['name']}.")
                self.tools_exhausted = True
        else:
            print(f"No more uses available for tools.")
            self.tools_exhausted = True

    def get_remaining_cooldown(self):
        """Return the remaining cooldown times for all tools."""
        current_time = time.time()
        remaining_cooldowns = []
        for tool in self.tools:
            time_since_last_use = current_time - tool['last_used']
            remaining_time = max(0, tool['cooldown'] - time_since_last_use)
            remaining_cooldowns.append(remaining_time)
            if remaining_time > 0:
                print(f"{tool['name']} has {remaining_time:.2f} seconds remaining on cooldown.")
        return remaining_cooldowns
