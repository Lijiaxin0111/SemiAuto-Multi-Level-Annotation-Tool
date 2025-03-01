def get_ancient_phase(phase_stroge, phase, ancient_lst=[]):

    if type(phase_stroge) == (list):
        if phase in phase_stroge:
            return ancient_lst
        return []

    if type(phase_stroge) == (dict):
        if phase in phase_stroge:
            return ancient_lst
        ans = []
        for key in phase_stroge:
            ans += get_ancient_phase(phase_stroge[key], phase, ancient_lst + [key])
        return ans


def get_child_phase(phase_stroge, phase, is_child=False):

    if is_child and type(phase_stroge) == (list):
        return phase_stroge

    if is_child and type(phase_stroge) == (dict):
        ans = []
        for key in phase_stroge:
            ans += [key]
            ans += get_child_phase(phase_stroge[key], phase, is_child=True)

        return ans

    if not is_child and type(phase_stroge) == (dict):
        if phase in phase_stroge:

            return [phase] + get_child_phase(phase_stroge[phase], phase, is_child=True)
        else:
            ans = []
            for key in phase_stroge:
                ans += get_child_phase(phase_stroge[key], phase, is_child=False)
            return ans

    if not is_child and type(phase_stroge) == (list):
        return []




def get_full_transition(phase_transition_dict, phase_stroge):
    ans = {}
    for key, value in phase_transition_dict.items():
        before_state = key.split("-")[0]
        after_state = key.split("-")[1]

        all_before_state = (
            get_child_phase(phase_stroge, before_state, is_child=False)
            + get_ancient_phase(phase_stroge, before_state)
            + [before_state]
        )
        all_after_state = (
            get_child_phase(phase_stroge, after_state, is_child=False)
            + get_ancient_phase(phase_stroge, after_state)
            + [after_state]
        )

        for all_before in all_before_state:
            for all_after in all_after_state:
                ans[all_before + "-" + all_after] = (
                    ans.get(all_before + "-" + all_after, []) + value
                )

    return ans


if __name__ == "__main__":

    phase_stroge = {
        "solid": {
            "particulate": [],
            "non_particulate": ["rigid body", "flexible body"],
        },
        "liquid": ["viscous", "non viscous"],
        "aerosol/gas": [],
    }


    phase_transition = {
        "particulate-particulate": ["split"],
        "rigid body-rigid body": ["separate", "merge"],
        "flexible body-flexible body": ["twist", "stretch"],
        "viscous-viscous": ["stretch", "paint"],
        "non viscous-non viscous": ["flow", "split", "paint", "mix"],
        "aerosol/gas-aerosol/gas": ["diffusion"],
        "solid-liquid": ["melt", "dissolve"],
        "liquid-solid": ["solidify"],
        "aerosol/gas-liquid": ["condense"],
        "liquid-aerosol/gas": ["vaporize"],
        "solid-aerosol/gas": ["sublimate"],
        "aerosol/gas-solid": ["deposition"],
        "non_particulate-particulate": ["break"],
        "rigid body-flexible body": ["soften"],
    }
    print(get_full_transition(phase_transition, phase_stroge))
