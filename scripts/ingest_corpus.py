import asyncio

from tahoe_operator.corpus import CorpusService


async def main() -> None:
    result = await CorpusService().index_documents()
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
