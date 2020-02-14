import matplotlib.pyplot as plt


class DrawTree(object):
    def __init__(self, root_node):
        self.node = root_node
        self.layer_dict = {}
        self.depth_count = []
        self.plot_count = []
        self.left_contour = []
        # Setup Functions:
        self.fill_depth_count_array(self.node)

    def fill_depth_count_array(self, node):
        # Check if we haven't been here before, append with 1 if so!
        if len(self.depth_count) < node.depth + 1:
            self.depth_count.append(1)
            self.plot_count.append(1)
        else:
            self.depth_count[node.depth] += 1
            self.plot_count[node.depth] += 1
        # Recursive call for children:
        for child in node.children_nodes:
            self.fill_depth_count_array(child)
        return

    def set_contour(self, node):
        if len(self.left_contour) < node.depth + 1:
            self.left_contour.append(node.x)
        elif self.left_contour[node.depth] > node.x:
            self.left_contour[node.depth] = node.x
        for child in node.children_nodes:
            self.set_contour(child)

    def plot_node(self, node):
        # Plot relevant lines to children nodes (if any)
        for child in node.children_nodes:
            self.plot_node(child)
            plt.plot([node.x, child.x], [-node.depth, -child.depth])
        # Plot this node as a point:
        plt.plot(node.x, -node.depth, 'k*')
        # Plot this node as a box with info:
        plt.annotate(node, (node.x, -node.depth + 1 / 8),
                     bbox=dict(facecolor='wheat', alpha=0.5, boxstyle='circle'),
                     fontsize=5)
        return

    def first_walk(self, node, spacing):
        if node.isLeaf:
            node.x = -(self.depth_count[node.depth] // 2 - self.plot_count[node.depth]) * spacing
            # Update the dictionary for this depth:
            self.plot_count[node.depth] -= 1
        else:
            # Calculate appropriate value of x
            x_children = []
            # Recursive part to plot line to next child node if any:
            for child in node.children_nodes:
                x_child = self.first_walk(child, spacing)
                x_children.append(x_child)
            node.x = -(self.depth_count[node.depth] // 2 - self.plot_count[node.depth]) * spacing
            self.plot_count[node.depth] -= 1

        return node.x

    def second_walk(self, node, min_spacing):
        if not node.isLeaf and all(child.isLeaf for child in node.children_nodes):
            return
        else:
            if node.children_nodes[0].x == self.left_contour[node.children_nodes[0].depth]:
                num_children = len(node.children_nodes)
                new_pos = node.x - (num_children * min_spacing) // 2
                for i in range(0, num_children):
                    self.move_right(node.children_nodes[i], node.children_nodes[i].x - new_pos)
                    self.second_walk(node.children_nodes[i], min_spacing)
                    new_pos += min_spacing
                self.left_contour[node.children_nodes[0].depth] = node.children_nodes[-1].x
        return

    def move_right(self, branch, n):
        branch.x += n
        for child in branch.children_nodes:
            self.move_right(child, n)

    def move_left(self, branch, n):
        branch.x -= n
        for child in branch.children_nodes:
            self.move_right(child, n)

    def print_tree(self, spacing):
        self.first_walk(self.node, spacing)
        self.set_contour(self.node)
        print(self.left_contour)
        plt.figure()
        self.plot_node(self.node)
        plt.show()

