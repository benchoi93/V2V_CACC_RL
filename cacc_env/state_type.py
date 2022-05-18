from .vehicle import Vehicle


def get_position(vehicle: Vehicle):
    return vehicle.x


def get_speed(vehicle: Vehicle):
    return vehicle.v


def get_acceleration(vehicle: Vehicle):
    return vehicle.a


def get_spacing(vehicle: Vehicle):
    return vehicle.leader.x - vehicle.x


def get_relative_speed(vehicle: Vehicle):
    return vehicle.v - vehicle.leader.v


def get_leader_spacing(vehicle: Vehicle):
    if vehicle.leader.leader is not None:
        return get_spacing(vehicle.leader)
    else:
        return 1e10


def get_leader_relative_speed(vehicle: Vehicle):
    if vehicle.leader.leader is not None:
        return get_relative_speed(vehicle.leader)
    else:
        return 120/3.6


def get_communication(vehicle: Vehicle):
    return vehicle.incoming_message


state_function = {
    "position": get_position,
    "speed": get_speed,
    "acceleration": get_acceleration,
    "spacing": get_spacing,
    "relative_speed": get_relative_speed,
    "leader_spacing": get_leader_spacing,
    "leader_relative_speed": get_leader_relative_speed,
    "message": get_communication
}

state_minmax_lookup = {
    "position": [0, 1e10],
    "speed": [0, 120/3.6],
    "acceleration": [-5, 5],
    "spacing": [0, 1e10],
    "relative_speed": [-120/3.6, 120/3.6],
    "leader_spacing": [0, 1e10],
    "leader_relative_speed": [-120/3.6, 120/3.6],
    "message": [-1, 1]
}
