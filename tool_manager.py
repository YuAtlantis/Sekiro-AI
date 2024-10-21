import time
from input_keys import perform_action


class ToolManager:
    def __init__(self):
        self.tools = ['Tool 1', 'Tool 2', 'Tool 3']
        self.current_tool_index = 0
        self.remaining_uses = 19
        self.cooldown = 4
        self.last_tool_use_time = 0
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
            current_time = time.time()
            if current_time - self.last_tool_use_time >= self.cooldown:
                perform_action("3", 0.1)
                print(f"Using {self.tools[self.current_tool_index]}.")
                self.remaining_uses -= 1
                print(f"Remaining uses for tools: {self.remaining_uses}")
            else:
                print(f"{self.tools[self.current_tool_index]} not cooldown right now")
        else:
            print(f"No more uses available for {self.tools[self.current_tool_index]}")
            self.tools_exhausted = True
