import json
import pickle
import random
from typing import List, Dict, Any
import aiofiles
import asyncio
from tqdm.asyncio import tqdm_asyncio


async def load_data(file_path: str, samples: int = 1, test: bool = False) -> List[Dict[str, Any]]:
    """
    Load trading data from a pickle file.

    Args:
        file_path (str): Path to the pickle file.
        samples (int): Number of samples to load.
        test (bool): Flag to indicate loading test data.

    Returns:
        List[Dict[str, Any]]: Loaded trading data.
    """
    async with aiofiles.open(file_path, 'rb') as file:
        content = await file.read()
        data = pickle.loads(content)

    if test:
        # Implement test data selection logic if applicable
        selected_data = data[:samples]
    else:
        selected_data = random.sample(data, samples) if samples < len(data) else data

    return selected_data
