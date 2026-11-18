"""Rebuild the DogMove audit and numerical cache used by the locked split."""

import argparse

from pet_imu.dogmove import audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit",))
    parser.parse_args()
    audit()


if __name__ == "__main__":
    main()
