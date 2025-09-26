from mesa import Agent, Model
from mesa.discrete_space import CellAgent
import random
import seaborn as sns

from typing import Literal

import math

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.discrete_space import OrthogonalMooreGrid
from mesa.visualization import SolaraViz, make_plot_component, SpaceRenderer
from mesa.visualization.components import AgentPortrayalStyle


def count_agents_by_type(model, agent_type):
    return len(model.agents_by_type[agent_type])

class Cell(CellAgent):
    """The base animal class."""

    def __init__(
        self, model, death_rate,  growth_rate, cell=None
    ):
        """Initialize an animal.

        Args:
            model: Model instance
            energy: Starting amount of energy
            p_reproduce: Probability of reproduction (asexual)
            energy_from_food: Energy obtained from 1 unit of food
            cell: Cell in which the animal starts
        """
        super().__init__(model)
        self.death_rate = death_rate
        self.growth_rate = growth_rate
        self.cell = cell
        self.alive = True
        # self.pos = self.cell.pos

    def find_neighbors(self, type: Literal['cancer', 'empty', 'healthy']):
        """Find neighbors of a specific type."""
        if type == 'cancer':
            neighbors = self.cell.get_neighborhood(radius=1, include_center=False)
            return [a for a in neighbors if any(isinstance(ag, CancerCell) for ag in a.agents)]
        elif type == 'healthy':
            neighbors = self.cell.get_neighborhood(radius=1, include_center=False)
            return [a for a in neighbors if any(isinstance(ag, HealthyCell) for ag in a.agents)]
        elif type == 'empty':
            neighborhood = self.cell.get_neighborhood(radius=1, include_center=False)
            return [pos for pos in neighborhood if pos.is_empty]
        
    
    def divide(self):
        """Create offspring, new instance."""
        if not self.alive:
            return
        if self._mesa_cell is None:
            return
        en = self.find_neighbors('empty')
        new_pos = random.choice(en) if en else None
        if new_pos and random.random() < self.growth_rate:
            self.create_agents(
                model=self.model,
                death_rate=self.death_rate,
                growth_rate=self.growth_rate,
                cell=new_pos,
                n=1
            )
        
    def cell_death(self):
        """"""
        self.alive = False
        self.remove()
    
    def move(self):
        """Move to a random empty neighboring cell."""
        neighbors = self.cell.get_neighborhood(radius=1, include_center=False)
        new_pos = neighbors.select_random_cell()
        self.cell = new_pos

class CancerCell(Cell):
    
    def necrosis(self):
        nb = self.find_neighbors('cancer')
        if len(nb) > 6:
            # If too many cancer cells around, this cell undergoes necrosis
            self.cell_death()
         
    def apoptosis(self):
        # If the cell is not dividing, it has a chance to undergo apoptosis
        if random.random() < self.death_rate:
            self.remove()
    
    def mutate(self):
        # Introduce a mutation with a small probability
        if random.random() < self.model.mutation_rate:
            self.growth_rate += random.uniform(-0.01, 0.01)
    
    def step(self):
        self.mutate()
        self.necrosis()
        self.apoptosis()
        # print(self.__dict__)
        self.divide()
        
class TCell(Cell):
    
    def attract(self):
        """Attract T-cells to cancer cells."""
        # Find all cancer cells in the neighborhood
        if random.random() < 0.5 and len(self.model.agents_by_type[self.__class__]) < 1000:
            self.create_agents(
                model=self.model,
                death_rate=self.death_rate,
                growth_rate=self.growth_rate,
                cell=self.random.choices(self.model.grid.all_cells.cells, k=1),
                n=1
            )
    
    def search(self):
        cnb = self.find_neighbors('cancer')
        if len(cnb) > 0:
            self.attract()
            for target in cnb:
                if random.random() < self.model.p_kill:
                    target.agents[0].cell_death() 
        else:
            self.move()
        
    def step(self):  # If there are cancer cells, T-cells can kill them
        self.search()
        self.search()             

class HealthyCell(Cell):
    
    def necrosis(self):
        cnb = self.find_neighbors('cancer')
        if random.random() < self.death_rate + (len(cnb) + 1)*0.05:
            self.cell_death()
        
    def apoptosis(self):
        # If the cell is not dividing, it has a chance to undergo apoptosis
        if random.random() < self.death_rate:
            self.cell_death()
    
    def step(self):
        # Check for necrosis
        self.necrosis()
        self.apoptosis()
        self.divide()
            

class CancerABM(Model):
    def __init__(self, 
                 width=100, 
                 height=100, 
                 init_cancer=1, 
                 init_Tcells=10, 
                 cancer_growth_rate=0.5,
                 tissue_turnover=0.2,
                 Tcell_kill_rate=0.9,
                 seed=42
                 ):
        super().__init__(seed=seed)
        self.grid = OrthogonalMooreGrid(
            (width, height), 
            torus=False,
            capacity=math.inf,
            random=self.random
            )
        # self.agents = AgentSet()
        
        self.p_kill = Tcell_kill_rate
        
        self.p_healthy_grow = tissue_turnover
        self.p_cancer_grow = cancer_growth_rate
        self.p_tcell_grow = 0
        
        self.p_healthy_cell_death = 0.01
        self.p_cancer_cell_death = 0.0001
        self.p_tcell_death = 0
        
        self.mutation_rate = 0.01  # Probability of mutation in cancer cells

        self.datacollector = DataCollector(
            model_reporters={
                "Cancer": lambda m: len(m.agents_by_type[CancerCell]),
                "Healthy": lambda m: len(m.agents_by_type[HealthyCell]),
                "TCells": lambda m: len(m.agents_by_type[TCell])
            }
        )
        
        
        HealthyCell.create_agents(
                self, 
                death_rate=self.p_healthy_cell_death,
                growth_rate=self.p_healthy_grow,
                cell = self.random.choices(self.grid.all_cells.cells, k=int(width*height*0.8)),
                n = int(width*height*0.8)
            )
        
        
        # Initialize cancer cells
        CancerCell.create_agents(
                self, 
                death_rate=self.p_cancer_cell_death,
                growth_rate=self.p_cancer_grow,
                cell = self.random.choices(self.grid.all_cells.cells, k=init_cancer),
                n = init_cancer
            )
            # self.grid.place_agent(cell, (x, y))
            # self.agents.add(cell)

        # Initialize T-cells
        TCell.create_agents(
                self, 
                death_rate=self.p_tcell_death,
                growth_rate=self.p_tcell_grow,
                cell = self.random.choices(self.grid.all_cells.cells, k=init_Tcells),
                n = init_Tcells
            )
            # self.grid.place_agent(cell, (x, y))
            # self.agents.add(cell)
        
    def step(self):    
        self.agents_by_type[CancerCell].do("step")
        self.agents_by_type[TCell].do("step")
        self.agents_by_type[HealthyCell].do("step")
        # self.agents.do("step")
        self.datacollector.collect(self)
        
# Visualization
def agent_portrayal(agent):
    if isinstance(agent, CancerCell):
        return AgentPortrayalStyle(color="#0F0A26", size=20)
    elif isinstance(agent, HealthyCell):
        return AgentPortrayalStyle(color="#33C7CC", size=20)
    elif isinstance(agent, TCell):
        return AgentPortrayalStyle(color="#FF6666", size=20)
    else:
        return AgentPortrayalStyle(color="white", size=20)


model_params = {
    "init_Tcells": {
        "type": "SliderInt",
        "value": 5,
        "label": "Number of T-cells:",
        "min": 1,
        "max": 25,
        "step": 1,
    },
    "cancer_growth_rate": {
        'type': 'SliderFloat',
        'value': 0.5,
        'label': 'Cancer Growth Rate',
        'min': 0.1,
        'max': 1.0,
        'step': 0.1,
        },
    "Tcell_kill_rate": {
        'type': 'SliderFloat',
        'value': 0.9,
        'label': 'T-cell Kill Rate',
        'min': 0.1,
        'max': 1.0,
        'step': 0.1,
        },
    'tissue_turnover': {
        'type': 'SliderFloat',
        'value': 0.2,
        'label': 'Tissue Turnover Rate',
        'min': 0.01,
        'max': 0.5,
        'step': 0.1,
        },
    "width": {
        "type": "SliderInt",
        "value": 100,
        "label": "Grid Width:",
        "min": 10,
        "max": 250,
        "step": 10,
    },
    "height": {
        "type": "SliderInt",
        "value": 100,
        "label": "Grid Height:",
        "min": 10,
        "max": 250,
        "step": 10,
    },
}

def main():
    model = CancerABM()
    # for _ in range(100):
    #     model.step()

    renderer = SpaceRenderer(model=model, backend="matplotlib").render(agent_portrayal=agent_portrayal)
    line_plot = make_plot_component(["Cancer", "Healthy", "TCells"], page=0)

    page = SolaraViz(
        model=model,
        renderer=renderer,
        model_params=model_params,
        components=[line_plot],
        title="Cancer-Immune System Interaction Model",
        description=(
            "A simple model of cancer growth and immune system interaction using cellular automata."
        ),
    )
    
    return page


if __name__ == "__main__":
    page = main()