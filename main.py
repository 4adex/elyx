import asyncio
from src import OrchestatedAgent


async def main():
    agent = OrchestatedAgent()
    sample = "I've been having persistent headaches for the past week, especially in the morning. They seem to get worse when I stand up quickly."
    result = await agent.start_conversation(sample)


if __name__ == "__main__":
    asyncio.run(main())