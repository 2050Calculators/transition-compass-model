# Define a class for handling sector connections
class Interface:
    def __init__(self):
        self.link = {}  # Store the connections in a dictionary with tuple keys

    def add_link(self, from_sector, to_sector, dm):
        key = (from_sector, to_sector)

        # Prevent reverse direction connections
        reverse_key = (to_sector, from_sector)
        if reverse_key in self.link:
            raise ValueError(f"Connection from {to_sector} to {from_sector} already exists. Can't add reverse.")

        # Add the new connection
        self.link[key] = dm

    def get_link(self, from_sector, to_sector):
        key = (from_sector, to_sector)
        if key in self.link:
            return self.link[key]
        else:
            raise KeyError(f"No connection from {from_sector} to {to_sector}.")

    def list_link(self):
        return list(self.link.keys())

    def has_link(self, from_sector, to_sector):
        key = (from_sector, to_sector)
        if key in self.link:
            return True
        else:
            return False