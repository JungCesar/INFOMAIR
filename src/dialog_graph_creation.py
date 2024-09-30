"""Part 1b"""
import graphviz
from IPython.display import Image, display

# create a new graph
diagram = graphviz.Digraph('Restaurant Recommendation System', format='png')
diagram.attr(rankdir='TB', nodesep='4', ranksep='0.2')

# define the nodes
diagram.attr('node', shape='box')
diagram.node('1', '1. welcome')
diagram.node('2', '2. ask area')
diagram.node('3', '3. ask food type')
diagram.node('4', '4. ask price range')
diagram.node('5', '5. check restaurant availability')
diagram.node('6', '6. offer alternatives')
diagram.node('7', '7. provide restaurant details')
diagram.node('8', 'End Conversation')

# define the expressions
diagram.attr('node', shape="ellipse", style='filled', fillcolor="lightpink")
diagram.node('i','express preferences')
diagram.node('ii','reply area')
diagram.node('iii','reply food type')
diagram.node('iv','reply price range')
diagram.node('v', 'user provides alternative preferences')

# define tussenstadia
diagram.attr('node', shape="diamond", style='filled', fillcolor='lightblue')
diagram.node('a', 'area expressed?')
diagram.node('b','food kind expressed?')
diagram.node('c','price range expressed?')
diagram.node('d','restaurant available?')
diagram.node('e','user asks for details?')

# Define the transitions
diagram.edge('1', 'i')
diagram.edge('i','a')
diagram.edge('a','2', label="no")
diagram.edge('a','b',label="yes")
diagram.edge ('2','ii')
diagram.edge('ii','b')
diagram.edge('b','3', label="no")
diagram.edge('b','c',label="yes")
diagram.edge('3','iii')
diagram.edge('iii','c')
diagram.edge('c','4',label="no")
diagram.edge('c','5',label="yes")
diagram.edge('4','iv')
diagram.edge('iv','5')
diagram.edge('5','d')
diagram.edge('d','v',label='no')
diagram.edge('d','7',label="yes")
diagram.edge('v','6')
diagram.edge('6','7')
diagram.edge('7','e')
diagram.edge('e','7', label="yes")
diagram.edge('e','8',label='no')


img = diagram.pipe(format='png')  # Render diagram in memory
display(Image(img))  # Display image directly in the notebook
