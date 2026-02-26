from avro_mapping import build_avro_mapping
from get_data import avro_to_pkl

def main():
    build_avro_mapping()
    avro_to_pkl()

if __name__ == "__main__":
    main()
