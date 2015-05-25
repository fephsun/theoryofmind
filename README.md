# theoryofmind
This is code for running the computational experiments described in *insert paper here*.  The code implements an inverse planning algorithm, which infers the goals of an agent, given the agent's actions.  For these experiments, the agent is picking up objects in a 2D grid room.  There are two colors of objects (labeled as 1 and 2 in the code), and the agent has different preferences for each color.  The inverse planning algorithm takes in the layout of the room and the path of the agent, and returns a posterior distribution over how much the agent prefers each color.

## Running the code
You need Python 2.7 to run this code.  No other dependencies are needed.  Execute `runner.py` or `runFromData.py`.  The algorithm takes a few minutes to complete.


## Interpreting the results
You should see a graph that looks like this: 
!(./outPlot.png)

The x-axis represents the value the agent assigns to color 1.  The value of color 2 is one minus the value of color 1.  The y-axis shows the log-likelihood of that value assignment, given the agent's actions.  In this case, the agent is likely to prefer color 1 at least a little bit over color 2.