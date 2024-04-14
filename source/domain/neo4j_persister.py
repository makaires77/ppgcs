class Neo4jPersister:
    def __init__(self, uri, username, password):
        self._driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        self._driver.close()

    @staticmethod
    def convert_to_primitives(input_data):
        if input_data is None:
            return None
        
        if isinstance(input_data, dict):
            return {key: Neo4jPersister.convert_to_primitives(value) for key, value in input_data.items()}
        
        elif isinstance(input_data, list):
            return [Neo4jPersister.convert_to_primitives(item) for item in input_data]
        
        elif isinstance(input_data, str):
            if 'http://' in input_data or 'https://' in input_data:
                parts = input_data.split(" ")
                new_parts = [urllib.parse.quote(part) if part.startswith(('http://', 'https://')) else part for part in parts]
                return " ".join(new_parts)
            return input_data
        
        elif isinstance(input_data, (int, float, bool)):
            return input_data
        
        else:
            return str(input_data)

    @staticmethod
    def debug_and_convert(input_data):
        try:
            return Neo4jPersister.convert_to_primitives(input_data)
        except:
            print("Conversion failed for:", input_data)
            raise

    def extract_lattes_id(self, infpes_list):
        """Extracts the Lattes ID from the InfPes list."""
        for entry in infpes_list:
            if 'ID Lattes:' in entry:
                # Extracting the numeric portion of the 'ID Lattes:' entry
                return entry.split(":")[1].strip()
        return None

    def persist_data(self, data_dict, label):
        data_dict_primitives = self.convert_to_primitives(data_dict)

        # Extracting the Lattes ID from the provided structure
        lattes_id = self.extract_lattes_id(data_dict.get("InfPes", []))
        
        if not lattes_id:
            print("Lattes ID not found or invalid.")
            return
        
        # Flatten the "Identificação" properties into the main dictionary
        if "Identificação" in data_dict_primitives:
            id_properties = data_dict_primitives.pop("Identificação")
            
            if isinstance(id_properties, dict):
                for key, value in id_properties.items():
                    # Adding a prefix to avoid potential property name conflicts
                    data_dict_primitives[f"Identificação_{key}"] = value
            else:
                # If it's not a dictionary, then perhaps store it as a single property (optional)
                data_dict_primitives["Identificação_value"] = id_properties

        with self._driver.session() as session:
            query = f"MERGE (node:{label} {{lattes_id: $lattes_id}}) SET node = $props"
            session.run(query, lattes_id=lattes_id, props=data_dict_primitives)

    def update_data(self, node_id, data_dict):
        data_dict_primitives = self.convert_to_primitives(data_dict)
        with self._driver.session() as session:
            query = f"MATCH (node) WHERE id(node) = {node_id} SET node += $props"
            session.run(query, props=data_dict_primitives)

    def parse_area(self, area_string):
        """Parses the area string and returns a dictionary with the parsed fields."""
        parts = area_string.split(" / ")
        area_data = {}
        for part in parts:
            key, _, value = part.partition(":")
            area_data[key.strip()] = value.strip()
        return area_data

    def process_all_person_nodes(self):
        """Iterates over all Person nodes and persists secondary nodes and relationships."""
        with self._driver.session() as session:
            result = session.run("MATCH (p:Person) RETURN p.name AS name, p.`Áreas de atuação` AS areas")

            for record in result:
                person_name = record["name"]
                
                # Check if name or areas is None
                if person_name is None or record["areas"] is None:
                    print(f"Skipping record for name {person_name} due to missing name or areas.")
                    continue

                # Check if the areas data is already in dict form
                if isinstance(record["areas"], dict):
                    areas = record["areas"]
                else:
                    # Attempt to convert from a string representation (e.g., JSON)
                    try:
                        areas = json.loads(record["areas"])
                    except Exception as e:
                        print(f"Failed to parse areas for name {person_name}. Error: {e}")
                        continue
                
                self.persist_secondary_nodes(person_name, areas)