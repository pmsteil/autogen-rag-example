"""
This is a sample script to show how to use RAG(retreival augmented generation) user proxy agent.
It was originally written by Microsoft and found here:
https://github.com/microsoft/autogen/blob/main/notebook/agentchat_groupchat_RAG.ipynb

This script will show how to use a user proxy agent which can retrieve content
from an external data source such as a web hosted document.

First it will show how to use RAG user proxy agent in a group chat.
Then it will show how to use RAG user proxy agent in a normal chat.
Here are the detailed steps:
1. Create a user proxy agent which can retrieve content from RAG.
2. Create a group chat with multiple agents.
3. Create a group chat manager.
4. Initiate chat with the user proxy agent.
5. Start chatting with other agents.
"""

import chromadb

import autogen
from autogen import AssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent


#####################################################################
# MAIN GLOBAL VARIABLES
# (based on original code from Microsoft)
#####################################################################

# load config list from .env file, map api keys to models, and filter models.
CONFIG_LIST = autogen.config_list_from_dotenv(
    dotenv_file_path=".env",
    model_api_key_map={
        "gpt-4": "OPENAI_API_KEY",
        "vicuna": "HUGGING_FACE_API_KEY",
    },
    filter_dict={
        "model": [ "gpt-3.5-turbo", "gpt-35-turbo", "gpt-35-turbo-0613", "gpt-4", ]
    },
)

# llm_config is used to configure the LLM model.
LLM_CONFIG_JSON = {
    "timeout": 60,
    "cache_seed": 42,
    "config_list": CONFIG_LIST,
    "temperature": 0,
}



#####################################################################
# DEFINE HIGH LEVEL FUNCTIONS
#####################################################################

def rag_chat( agents, prompt: str ):
    """
    This is a sample script to show how to use RAG user proxy agent.
    Details:
    2. Create a group chat with multiple agents using the boss_aid instead of the boss agent.
    3. Create a group chat manager.
    4. Initiate chat with the user proxy agent.
    """
    _reset_agents( agents )
    boss_aid = agents['boss_aid']
    coder = agents['coder']
    pm = agents['pm']
    reviewer = agents['reviewer']

    groupchat = autogen.GroupChat(
        agents=[boss_aid, coder, pm, reviewer],
        messages=[],
        max_round=12,
        speaker_selection_method="round_robin"
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=LLM_CONFIG_JSON)

    # Start chatting with boss_aid as this is the user proxy agent.
    boss_aid.initiate_chat(
        manager,
        problem=prompt,
        n_results=3,
    )


def norag_chat( agents, prompt: str ):
    """
    This is a sample script to show how to use RAG user proxy agent without RAG (w/o the boss_aid agent).
    """
    _reset_agents( agents )
    boss = agents['boss']
    coder = agents['coder']
    pm = agents['pm']
    reviewer = agents['reviewer']

    groupchat = autogen.GroupChat(
        agents=[boss, coder, pm, reviewer],
        messages=[],
        max_round=12,
        speaker_selection_method="auto",
        allow_repeat_speaker=False,
    )
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=LLM_CONFIG_JSON)

    # Start chatting with the boss as this is the user proxy agent.
    boss.initiate_chat(
        manager,
        message=prompt,
    )





def call_rag_chat( agents: list, prompt: str ):
    """
    In this case, we will have multiple user proxy agents and we don't initiate the chat
    with RAG user proxy agent.
    In order to use RAG user proxy agent, we need to wrap RAG agents in a function and call
    it from other agents.
    """
    _reset_agents( agents )
    boss        = agents['boss']
    boss_aid    = agents['boss_aid']
    coder       = agents['coder']
    pm          = agents['pm']
    reviewer    = agents['reviewer']

    # This function will be used to retrieve content from RAG.
    def retrieve_content(message, n_results=3):
        boss_aid.n_results = n_results  # Set the number of results to be retrieved.

        # Check if we need to update the context - this means that the user has asked a question and we need to retrieve content for it?
        update_context_case1, update_context_case2 = boss_aid._check_update_context(message)

        # If we need to update the context, then we need to retrieve content for it.
        if (update_context_case1 or update_context_case2) and boss_aid.update_context:
            boss_aid.problem = message if not hasattr(boss_aid, "problem") else boss_aid.problem
            _, ret_msg = boss_aid._generate_retrieve_user_reply(message)
        else:
            # Otherwise, we need to generate an initial message.
            ret_msg = boss_aid.generate_init_message(message, n_results=n_results)
        return ret_msg if ret_msg else message

    boss_aid.human_input_mode = "NEVER"  # Disable human input for boss_aid since it only retrieves content.

    # llm_config2_json is used to configure the LLM model for the user proxy agent.
    llm_config2_json = {
        "functions": [
            {
                "name": "retrieve_content",
                "description": "retrieve content for code generation and question answering.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Refined message which keeps the original meaning and can be used to retrieve content for code generation and question answering.",
                        }
                    },
                    "required": ["message"],
                },
            },
        ],
        "config_list": CONFIG_LIST,
        "timeout": 60,
        "cache_seed": 42,
    }

    for agent in [coder, pm, reviewer]:
        # update llm_config for assistant agents.
        agent.llm_config.update(llm_config2_json)

    for agent in [boss, coder, pm, reviewer]:
        # register functions for all agents.
        agent.register_function(
            function_map={
                "retrieve_content": retrieve_content,
            }
        )

    groupchat = autogen.GroupChat(
        agents=[boss, coder, pm, reviewer],
        messages=[],
        max_round=12,
        speaker_selection_method="random",
        allow_repeat_speaker=False,
    )

    manager_llm_config = llm_config2_json.copy()
    manager_llm_config.pop("functions")
    manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=manager_llm_config)

    # Start chatting with the boss as this is the user proxy agent.
    boss.initiate_chat(
        manager,
        message=prompt,
    )


#####################################################################
# UTILITY FUNCTIONS
#####################################################################
def termination_msg(_msg):
    """
    Check if the message is a termination message.
    """
    return isinstance(_msg, dict) and "TERMINATE" == str(_msg.get("content", ""))[-9:].upper()

def _reset_agents( agents: list ):
    """
    Reset all agents.
    """
    # loop over each key in the agents dict and reset the agent.
    for key in agents:
        agents[key].reset()


def _print_agents( agents: list ):
    """
    Reset all agents.
    """
    # loop over each key in the agents dict and print the agent's name and system message.
    for key in agents:
        print( f"agent['{key}']: ", agents[key] )









def create_agents():
    """
    Create all the agents we need.
    """


    # Create a boss agent which will ask questions and give tasks.
    boss = autogen.UserProxyAgent(
        name="Boss",
        is_termination_msg=termination_msg,
        human_input_mode="NEVER",
        system_message="The boss who ask questions and give tasks.",
        code_execution_config=False,  # we don't want to execute code in this case.
        default_auto_reply="Reply `TERMINATE` if the task is done.",
    )

    # Create a boss assistant agent which will retrieve content from RAG, namely a web hosted readme file.
    # The RetrieveUserProxyAgent is a user proxy agent which can retrieve content from RAG (retrieveal augmented generation).
    boss_aid = RetrieveUserProxyAgent(
        name="Boss_Assistant",
        is_termination_msg=termination_msg,
        system_message="Assistant who has extra content retrieval power for solving difficult problems.",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=3,
        retrieve_config={
            "task": "code",
            "docs_path": "https://raw.githubusercontent.com/microsoft/FLAML/main/website/docs/Examples/Integrate%20-%20Spark.md",
            "chunk_token_size": 1000,
            "model": CONFIG_LIST[0]["model"],
            "client": chromadb.PersistentClient(path="./tmp/chromadb"),
            "collection_name": "groupchat",
            "get_or_create": True,
        },
        code_execution_config=False,  # we don't want to execute code in this case.
    )

    # Create a coder agent which will write code.
    coder = AssistantAgent(
        name="Senior_Python_Engineer",
        is_termination_msg=termination_msg,
        system_message="You are a senior python engineer. Reply `TERMINATE` in the end when everything is done.",
        llm_config=LLM_CONFIG_JSON,
    )

    # Create a product manager agent which will manage the product.
    pm = autogen.AssistantAgent(
        name="Product_Manager",
        is_termination_msg=termination_msg,
        system_message="You are a product manager. Reply `TERMINATE` in the end when everything is done.",
        llm_config=LLM_CONFIG_JSON,
    )

    # Create a code reviewer agent which will review the code.
    reviewer = autogen.AssistantAgent(
        name="Code_Reviewer",
        is_termination_msg=termination_msg,
        system_message="You are a code reviewer. Reply `TERMINATE` in the end when everything is done.",
        llm_config=LLM_CONFIG_JSON,
    )

    return { "boss": boss, "boss_aid": boss_aid, "coder": coder, "pm": pm, "reviewer": reviewer }









def main():

    # define the problem to be solved.
    PROBLEM = "How to use spark for parallel training in FLAML? Give me sample code."

    print("LLM models to be used: ", [CONFIG_LIST[i]["model"] for i in range(len(CONFIG_LIST))])

    # create all the agents we need
    agents = create_agents()
    # _print_agents( agents )

    method = "3"
    # here are the three ways to use RAG user proxy agent.
    if method == "1":
        print( "method 1: norag_chat()" )
        norag_chat( agents, prompt=PROBLEM )  # this way does not use RAG.
    elif method == "2":
        print( "method 2: rag_chat()" )
        rag_chat( agents, prompt=PROBLEM )
    elif method == "3":
        print( "method 3: call_rag_chat()" )
        call_rag_chat( agents, prompt=PROBLEM )  # this way uses RAG and multiple user proxy agents.



if __name__ == "__main__":
    main()

