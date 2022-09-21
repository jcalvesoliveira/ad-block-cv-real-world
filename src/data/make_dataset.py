# -*- coding: utf-8 -*-
import os
import click
import logging
import pandas as pd
from pathlib import Path


def read_multiple_csvs(path: Path) -> pd.DataFrame:
    """Read multiple csv files into a single dataframe

    Args:
        path (Path): Path to the directory containing the csv files

    Returns:
        pandas.DataFrame: Dataframe containing all the csv files
    """
    dfs = []
    for file in os.listdir(path):
        if file.endswith(".csv"):
            new_df = pd.read_csv(path / file)
            new_df['filename'] = file[0:-4]
            dfs.append(new_df)
    return pd.concat(dfs)


@click.command()
@click.argument('input_folder', type=click.Path(exists=True))
@click.argument('output_folder', type=click.Path())
def main(input_folder, output_folder):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    annotations_dfs = read_multiple_csvs(Path(input_folder))
    annotations_dfs.to_csv(f'{output_folder}/annotations.csv', index=False)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
