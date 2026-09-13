"""Fixed teacher-only group energies; original54 cells use dataset/phase/group order."""
GROUP_NAMES = ('root_position', 'root_rotation', 'joint_position', 'root_linear_velocity', 'root_angular_velocity', 'joint_velocity')
ZERO_RESPONSE_ENERGIES = (3.268289691393298e-05, 0.0002707773000538789, 3.393011971879691e-05, 0.0004327436763933723, 0.0003535704853193714, 3.348514657479436e-05)
MEAN_ZERO_RESPONSE_ENERGY = 0.00019286493749569113
GROUP_WEIGHTS = (5.901096772528486, 0.7122640541039265, 5.684180872160203, 0.44567938947852637, 0.5454780461142876, 5.759716089786164)
CELL_GROUPS = tuple(range(6)) * 9
CONDITIONING_REPORT_SHA256 = '84a1fb497cdc0c6edf3b8db7be69a604dc4b1b7e73041b437f8c899d882f429c'
WEIGHT_RULE = 'mean_six_teacher_group_energies_over_group_energy'
