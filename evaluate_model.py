import os
from apted import APTED, PerEditOperationConfig

class Tree(object):
    """Represents a Tree Node"""

    def __init__(self, name, *children):
        self.name = name
        self.children = list(children)

    def bracket(self):
        """Show tree using brackets notation"""
        result = str(self.name)
        for child in self.children:
            result += child.bracket()
        return "{{{}}}".format(result)

    def __repr__(self):
        return self.bracket()

    @classmethod
    def from_text(cls, text):
        """Create tree from bracket notation

        Bracket notation encodes the trees with nested parentheses, for example,
        in tree {A{B{X}{Y}{F}}{C}} the root node has label A and two children
        with labels B and C. Node with label B has three children with labels
        X, Y, F.
        """
        tree_stack = []
        stack = []
        for letter in text:
            if letter == "{":
                stack.append("")
            elif letter == "}":
                text = stack.pop()
                children = deque()
                while tree_stack and tree_stack[-1][1] > len(stack):
                    child, _ = tree_stack.pop()
                    children.appendleft(child)

                tree_stack.append((cls(text, *children), len(stack)))
            else:
                stack[-1] += letter
        return tree_stack[0][0]
    

names = {}
obj_counter = 0

def build_tree(directory):
  """Builds a Tree structure representing the directory structure.

  Args:
      directory: The root directory path.

  Returns:
      A Tree object representing the directory structure.
  """
  global names, obj_counter
  root_node = Tree("root")  # Get root directory name
  for root, dirs, files in os.walk(directory):
    current_node = root_node
    for dir in root.split(os.sep)[1:]:  # Skip root directory path component
      child_node = next((child for child in current_node.children if child.name == dir), None)
      if not child_node:
        if dir in names:
            child_node = Tree(names[dir])
        else:
            child_node = Tree(str(obj_counter))
            names[dir] = str(obj_counter)
            obj_counter += 1
      current_node.children.append(child_node)
      current_node = child_node
    for file in files:
      if file in names:
        current_node.children.append(Tree(names[file]))
      else:
        current_node.children.append(Tree(str(obj_counter)))
        names[file] = str(obj_counter)
        obj_counter += 1
  return root_node


# Example usage:
directory_path = "/path/to/your/directory"
tree_str = build_tree(directory_path)


def compare_trees(tree1, tree2):
    apted = APTED(tree1, tree2)
    return apted.compute_edit_distance()

def main():
    original_tree = build_tree("./clustered_files")
    new_tree = build_tree("./unseen_files")
    print(compare_trees(original_tree, new_tree))

main()
