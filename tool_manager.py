from input_keys import perform_action


class ToolManager:
    def __init__(self):
        self.tools = ['Tool 1', 'Tool 2', 'Tool 3']
        self.current_tool_index = 0
        self.remaining_uses = 19
        self.tools_exhausted = False

    def change_tool(self):
        perform_action("Z", 0.1)
        self.current_tool_index = (self.current_tool_index + 1) % len(self.tools)
        print(f"Changed to {self.tools[self.current_tool_index]}.")

    def use_specific_tool(self, target_tool_index):
        while self.current_tool_index != target_tool_index:
            self.change_tool()
        self.use_tool()

    def use_tool(self):
        if self.remaining_uses > 0:
            perform_action("3", 0.1)
            print(f"Using {self.tools[self.current_tool_index]}.")
            self.remaining_uses -= 1
            print(f"Remaining uses for tools: {self.remaining_uses}")
        else:
            print("No more uses available for tools.")
            self.tools_exhausted = True