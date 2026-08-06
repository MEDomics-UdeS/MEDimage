class SUVHeaderProxy:
    """
    Mimics pydicom's dcm[tag].value and dcm[seq][0][tag].value syntax.
    """
    class Element:
        def __init__(self, value):
            self.value = value
        
        def __getitem__(self, idx):
            # Allows dcm[seq][0]
            if isinstance(self.value, list):
                return SUVHeaderProxy(self.value[idx])
            return self.value

        def __len__(self):
            return len(self.value) if isinstance(self.value, list) else 1

    def __init__(self, data_dict):
        self.data = data_dict if data_dict is not None else {}

    def __getitem__(self, tag):
        # Convert (0xGGGG, 0xEEEE) tuple to 0xGGGGEEEE integer
        if isinstance(tag, tuple):
            tag = (tag[0] << 16) | tag[1]
        
        val = self.data.get(tag)
        if val is None:
            raise KeyError(f"Tag {hex(tag) if isinstance(tag, int) else tag} not found")
        return self.Element(val)

    def __getattribute__(self, name):
        # Fallback for .value calls on the proxy itself (during recursion)
        if name == "value":
            return self.data
        return super().__getattribute__(name)

    def __contains__(self, tag):
        if isinstance(tag, tuple):
            tag = (tag[0] << 16) | tag[1]
        return tag in self.data
    
    def contains(self, tag):
        return self.__contains__(tag)

    def get(self, tag, default=None):
        return self.data.get(tag, default)
