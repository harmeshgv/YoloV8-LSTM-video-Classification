import json
import os

from src.config import STATE_FILE

def _read_state_file():
    with open(STATE_FILE, 'r') as file:
        return json.load(file)
    
def _write_state_file(state):
    with open(STATE_FILE , "w") as file:
        json.dump(state, file)

def get_person_id_counter():
    state = _read_state_file()
    return state.get("person_id_counter", 0)

def set_person_id_counter(person_id_counter):
    state = _read_state_file()
    state["person_id_counter"] = person_id_counter
    write_state_file(state)
