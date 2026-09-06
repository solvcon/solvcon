# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Download, extract, and load the NASA flight dataset.
"""

import dataclasses
import json
import ssl
import pathlib
import urllib.request
import zipfile

from . import dataframe

__all__ = ["NasaDataset", "EventReference"]


@dataclasses.dataclass(frozen=True)
class _EventDataView:
    """
    Lazy view of one row in a source-specific dataframe.

    :ivar dataframe: Source dataframe backing the row view.
    :vartype dataframe: DataFrame
    :ivar row: Row index in ``dataframe``.
    :vartype row: int
    """
    dataframe: 'dataframe.DataFrame' = dataclasses.field(
        repr=False,
        compare=False,
    )
    row: int

    def __getitem__(self, column):
        """
        Return the value of one column in the referenced row.

        :param column: Column name in the source dataframe.
        :type column: str
        :return: Scalar value stored at ``column`` and ``row``.
        :rtype: object
        """
        value = self.dataframe[column][self.row]
        return value

    def to_dict(self):
        """
        Materialize the referenced row as a dictionary.

        :return: Mapping from column name to row value.
        :rtype: dict[str, object]
        """
        return {
            column: self[column]
            for column in self.dataframe.columns
        }

    def __repr__(self):
        """
        Return a dictionary-like representation of the referenced row.

        :return: String form of the materialized row dictionary.
        :rtype: str
        """
        return repr(self.to_dict())


@dataclasses.dataclass(frozen=True)
class EventReference:
    """
    Reference one timestamped row in a source-specific dataset.

    :ivar dataset: Parent dataset that owns the referenced row.
    :vartype dataset: NasaDataset
    :ivar timestamp: Event timestamp in nanoseconds.
    :vartype timestamp: int
    :ivar source: Source dataset name.
    :vartype source: str
    :ivar row: Row index in the source dataframe.
    :vartype row: int
    """
    dataset: "NasaDataset" = dataclasses.field(repr=False, compare=False)
    timestamp: int
    source: str
    row: int

    @property
    def data(self):
        """
        Return a lazy view of the original row data.

        :return: Lazy row view backed by the source dataframe.
        :rtype: _EventDataView
        """
        return _EventDataView(self.dataset.dataframes[self.source], self.row)


class NasaDataset:
    """
    Helper for downloading, extracting, and loading NASA files.

    :ivar url: NASA API endpoint returning a presigned download URL.
    :vartype url: str
    :ivar download_dir: Local directory used for downloaded
        and extracted files.
    :vartype download_dir: pathlib.Path
    :ivar filename: Dataset zip filename.
    :vartype filename: str
    :ivar imu_csv: Path to the IMU CSV file.
    :vartype imu_csv: pathlib.Path or str
    :ivar lidar_csv: Path to the lidar CSV file.
    :vartype lidar_csv: pathlib.Path or str
    :ivar gt_csv: Path to the ground-truth CSV file.
    :vartype gt_csv: pathlib.Path or str
    :ivar dataframes: Loaded source dataframes keyed by source name.
    :vartype dataframes: dict[str, DataFrame]
    :ivar events: Timestamp-ordered event references.
    :vartype events: list[EventReference]
    """

    def __init__(self, url, filename):
        """
        Initialize download/load configuration.

        :param url: NASA API endpoint returning the presigned download URL.
        :type url: str
        :param filename: Name of the downloaded zip archive.
        :type filename: str
        :return: None
        :rtype: None
        """

        self.url = url
        self.download_dir = pathlib.Path.cwd() / ".cache" / "download"
        self.filename = filename
        self.csv_dir = self.download_dir / "Flight1_Catered_Dataset-20201013"
        self.csv_dir /= "Data"
        self.imu_csv = self.csv_dir / "dlc.csv"
        self.lidar_csv = self.csv_dir / "commercial_lidar.csv"
        self.gt_csv = self.csv_dir / "truth.csv"
        self.events: list[EventReference] = []
        self.dataframes: dict[str, dataframe.DataFrame] = {}

    def download(self):
        """
        Download zipped dataset from NASA presigned URL.

        :return: ``None``.
        :rtype: None
        """
        file_path = pathlib.Path(self.download_dir / self.filename)
        if file_path.exists():
            print(f"{file_path} exists,skip download.")
            return

        # The TechPort API rejects requests carrying Python's default
        # ``Python-urllib`` User-Agent (HTTP 403 Forbidden), so wrap the
        # call in a Request that advertises a project-identifying agent.
        req = urllib.request.Request(
            self.url,
            headers={"User-Agent": "solvcon-track/1.0"},
        )
        response = urllib.request.urlopen(req)
        presigned_url = json.loads(response.read())["presignedUrl"]
        self.download_dir.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(
            presigned_url,
            f"{self.download_dir}/{self.filename}",
            reporthook=self._download_hook,
        )

    def extract(self):
        """
        Extract downloaded dataset zip into ``download_dir``.

        :return: ``None``.
        :rtype: None
        """
        with zipfile.ZipFile(
            f"{self.download_dir}/{self.filename}",
            "r",
        ) as file:
            file.extractall(self.download_dir)

    def _download_hook(self, block_num, block_size, total_size):
        """
        Progress callback for ``urllib.request.urlretrieve``.

        :param block_num: Downloaded block count.
        :type block_num: int
        :param block_size: Block size in bytes.
        :type block_size: int
        :param total_size: Total file size in bytes.
        :type total_size: int
        :return: ``None``.
        :rtype: None
        """
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        ratio = min(downloaded / total_size, 1.0)
        width = 30
        filled = int(width * ratio)
        bar = "#" * filled + "-" * (width - filled)
        print(f"\rDownloading [{bar}] {ratio:6.2%}", end="", flush=True)

    def load(self):
        """
        Load all source datasets and build the timestamp timeline.

        :return: None
        :rtype: None
        """
        self.dataframes["imu"] = self._load_dataframe(self.imu_csv)
        self.dataframes["lidar"] = self._load_dataframe(self.lidar_csv)
        self.dataframes["ground_truth"] = self._load_dataframe(self.gt_csv)
        self._rebuild_timeline()

    def _load_dataframe(self, path):
        """
        Load one CSV file into a time-series dataframe.

        :param path: Path to a source CSV file.
        :type path: pathlib.Path or str
        :return: Loaded dataframe for the source file.
        :rtype: DataFrame
        """
        tsdf = dataframe.DataFrame()
        tsdf.read_from_text_file(
            path,
            delimiter=",",
            timestamp_column="TIME_NANOSECONDS_TAI",
        )
        return tsdf

    def _rebuild_timeline(self):
        """
        Rebuild the timestamp timeline from loaded source dataframes.

        :return: None
        :rtype: None
        """
        timeline_map: dict[int, list[EventReference]] = {}
        for source, df in self.dataframes.items():
            for row_index, timestamp in enumerate(df.index):
                timestamp = int(timestamp)
                timeline_map.setdefault(timestamp, []).append(
                    EventReference(
                        dataset=self,
                        timestamp=timestamp,
                        source=source,
                        row=row_index,
                    )
                )

        self.events = [
            ref
            for timestamp in sorted(timeline_map)
            for ref in timeline_map[timestamp]
        ]

    def __len__(self):
        """
        Return the number of events in the timeline.

        :return: Number of loaded events.
        :rtype: int
        """
        return len(self.events)

    def __getitem__(self, idx):
        """
        Return the event reference at ``idx``.

        :param idx: Event index.
        :type idx: int
        :return: Event reference at the specified index.
        :rtype: EventReference
        """
        return self.events[idx]


def main():
    ssl._create_default_https_context = ssl._create_stdlib_context
    dataset = NasaDataset(
        "https://techport.nasa.gov/api/file/presignedUrl/380503",
        "DDL-F1_Dataset-20201013.zip",
    )
    dataset.download()
    dataset.extract()


if __name__ == "__main__":
    main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
