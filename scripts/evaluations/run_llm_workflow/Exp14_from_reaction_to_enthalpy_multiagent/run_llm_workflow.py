import json
from chemgraph.agent.llm_agent import ChemGraph
from chemgraph.utils.get_workflow_from_llm import get_workflow_from_state
import argparse
import datetime


def get_query(
    reaction: dict,
    query_name: str = "enthalpy",
    temperature: float = 298,
    pressure: float = 101325,
    method: str = "mace_mp",
) -> str:
    """Get query for LLM.

    Returns:
        _type_: _description_
    """
    reactants_str = " + ".join([f"{r['coefficient']} ({r['name']})" for r in reaction["reactants"]])
    products_str = " + ".join([f"{p['coefficient']} ({p['name']})" for p in reaction["products"]])

    reaction_equation = f"{reactants_str} -> {products_str}"
    query_dict = {
        "enthalpy": f"Calculate the reaction enthalpy for this reaction: {reaction_equation}",
        "enthalpy_method": f"You are given a chemical reaction: {reaction_equation}. Calculate the enthalpy for this reaction using {method} at {temperature}K.",
        "gibbs_free_energy": f"What is the Gibbs free energy of reaction for {reaction_equation}?",
        "gibbs_free_energy_method": f"What is the Gibbs free energy of reaction for {reaction_equation} using {method}?",
        "gibbs_free_energy_method_temperature": f"What is the Gibbs free energy of reaction for {reaction_equation} using {method} at {temperature}K?",
    }

    return query_dict.get(query_name, "Query not found")  # Returns the query or a default message


def main(n_reactions: int):
    """ """
    # Load SMILES data from the specified JSON file
    combined_data = {}

    cca = ChemGraph(
        model_name='gpt-4o-mini',
        workflow_type="multi_agent",
        structured_output=True,
        return_option="state",
    )
    with open("reaction_dataset.json", "r") as rf:
        reactions = json.load(rf)

    # Iterate through the first n_structures molecules
    for idx, reaction in enumerate(reactions[:n_reactions]):
        print("********************************************")
        print(
            f"REACTION INDEX {reaction['reaction_index']}: REACTION NAME: {reaction['reaction_name']}"
        )
        print("********************************************")

        name = reaction["reaction_name"]

        query = get_query(
            reaction, query_name="enthalpy_method", method="GFN2-xTB", temperature=400
        )

        try:
            state = cca.run(query, config={"configurable": {"thread_id": f"{str(idx)}"}})
        except Exception as e:
            print(e)

        llm_workflow = get_workflow_from_state(state)

        # Store results in a structured dictionary
        state_data = cca.write_state(config={"configurable": {"thread_id": f"{str(idx)}"}})

        combined_data[name] = {"llm_workflow": llm_workflow}
        combined_data[name]["metadata"] = state_data

    # Save the results to a JSON file
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"llm_workflow_{timestamp}.json"

    # Save the results to a JSON file
    with open(filename, "w") as f:
        json.dump(combined_data, f, indent=4)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Calculate properties of a reaction.")
    parser.add_argument(
        "--n_reactions", type=int, default=10, help="Number of molecules to process (default: 10)"
    )
    args = parser.parse_args()

    # Call the main function with parsed arguments
    main(args.n_reactions)
