class PassportPhotoSpec:
    def __init__(self, width, height, background_color):
        self.width = width
        self.height = height
        self.background_color = background_color
        self.aspect = width / height
        self.chin_to_eyes = self.set_chin_eye_distance()

        if self.chin_to_eyes is None:
            raise NotImplementedError(
                "Please Implement country spec and overload get_chin_to_eyes"
            )

    def set_chin_eye_distance(self):
        return None

    def print_spec(self):
        raise NotImplementedError()


class GermanPassportPhotoSpec(PassportPhotoSpec):
    def set_chin_eye_distance(self):
        return 16.1

    def print_spec(self):
        print("German Passport Photo Specification")
        print(f"Width: {self.width} mm")
        print(f"Height: {self.height} mm")
        print(f"Background Color: {self.background_color}")


class AmericanPassportPhotoSpec(PassportPhotoSpec):
    def set_chin_eye_distance(self):
        return 1 / 2 - 7 / 48  # todo verify this

    def print_spec(self):
        print("American Passport Photo Specification")
        print(f"Width: {self.width} inches")
        print(f"Height: {self.height} inches")
        print(f"Background Color: {self.background_color}")


class PassportPhotoSpecFactory:
    def __new__(cls, country: str):
        if country == "Germany":
            return GermanPassportPhotoSpec(35, 45, "grey")
        elif country == "USA":
            return AmericanPassportPhotoSpec(2, 2, "white")
        else:
            raise ValueError("Unsupported country")


if __name__ == "__main__":
    c = PassportPhotoSpecFactory("Germany")
    c.print_spec()
