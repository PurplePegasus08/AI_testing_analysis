from pandasai import Agent
from pandasai_litellm.litellm import LiteLLM
import pandas as pd
llm = LiteLLM(model="gemini/gemini-2.5-flash", api_key="AIzaSyB0pmTvTBZ_HHVmG9ujEB9PthfVWeulwX4")
df = pd.read_csv('train.csv')

agent = Agent(df,
              config={"llm":llm,
                      "enforce_privacy":True,
                      "use_sql":False,
                      "custom_whitelisted_dependencies":["pandas","numpy","plotly"],
                      "enable_cache":True})

response = agent.chat("Generate charts using like in professional dashboard style like tablue or powerbi which give insight about data and tells the story in html. ")
print(response)


