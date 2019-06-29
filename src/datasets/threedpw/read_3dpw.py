"""
Reads 3DPW dataset.
"""

def get_3dpw2coco():
    # Conversion from 3DPW to coco universal (25)
    joint_names_coco = [
        'R Heel',
        'R Knee',
        'R Hip',
        'L Hip',
        'L Knee',
        'L Heel',
        'R Wrist',
        'R Elbow',
        'R Shoulder',
        'L Shoulder',
        'L Elbow',
        'L Wrist',
        'Neck',
        'Head',
        'Nose',
        'L Eye',
        'R Eye',
        'L Ear',
        'R Ear',
        'L Big Toe',
        'R Big Toe',
        'L Small Toe',
        'R Small Toe',
        'L Ankle',
        'R Ankle',
    ]

    joint_names_3dpw = [
        'Nose',
        'Neck',
        'R Shoulder',
        'R Elbow',
        'R Wrist',
        'L Shoulder',
        'L Elbow',
        'L Wrist',
        'R Hip',
        'R Knee',
        'R Ankle',
        'L Hip',
        'L Knee',
        'L Ankle',
        'R Eye',
        'L Eye',
        'R Ear',
        'L Ear',
        # Below are the missing parts        
        'Head',
        'L Big Toe',
        'R Big Toe',
        'L Small Toe',
        'R Small Toe',
        'L Heel',
        'R Heel',
    ]

    indices = [joint_names_3dpw.index(name) for name in joint_names_coco]

    return indices, joint_names_coco


def get_sequences(data_dir, split):
    test_seqs = [
        'downtown_arguing_00',
        'downtown_bar_00',
        'downtown_bus_00',
        'downtown_cafe_00',
        'downtown_cafe_01',
        'downtown_car_00',
        'downtown_crossStreets_00',
        'downtown_downstairs_00',
        'downtown_enterShop_00',
        'downtown_rampAndStairs_00',
        'downtown_runForBus_00',
        'downtown_runForBus_01',
        'downtown_sitOnStairs_00',
        'downtown_stairs_00',
        'downtown_upstairs_00',
        'downtown_walkBridge_01',
        'downtown_walking_00',
        'downtown_walkUphill_00',
        'downtown_warmWelcome_00',
        'downtown_weeklyMarket_00',
        'downtown_windowShopping_00',
        'flat_guitar_01',
        'flat_packBags_00',
        'office_phoneCall_00',
        'outdoors_fencing_01',
    ]
    train_seqs = [
        'courtyard_arguing_00',
        'courtyard_backpack_00',
        'courtyard_basketball_00',
        'courtyard_bodyScannerMotions_00',
        'courtyard_box_00',
        'courtyard_capoeira_00',
        'courtyard_captureSelfies_00',
        'courtyard_dancing_01',
        'courtyard_giveDirections_00',
        'courtyard_golf_00',
        'courtyard_goodNews_00',
        'courtyard_jacket_00',
        'courtyard_laceShoe_00',
        'courtyard_rangeOfMotions_00',
        'courtyard_relaxOnBench_00',
        'courtyard_relaxOnBench_01',
        'courtyard_shakeHands_00',
        'courtyard_warmWelcome_00',
        'outdoors_climbing_00',
        'outdoors_climbing_01',
        'outdoors_climbing_02',
        'outdoors_freestyle_00',
        'outdoors_slalom_00',
        'outdoors_slalom_01',
    ]
    val_seqs = [
        'courtyard_basketball_01',
        'courtyard_dancing_00',
        'courtyard_drinking_00',
        'courtyard_hug_00',
        'courtyard_jumpBench_01',
        'courtyard_rangeOfMotions_01',
        'downtown_walkDownhill_00',
        'outdoors_crosscountry_00',
        'outdoors_freestyle_01',
        'outdoors_golf_00',
        'outdoors_parcours_00',
        'outdoors_parcours_01',
    ]

    if split == 'val':
        return sorted(val_seqs)
    elif split == 'test':
        return sorted(test_seqs)
    elif split == 'train':
        return sorted(train_seqs)
    else:
        import ipdb
        ipdb.set_trace()
        print('split {} unknown'.format(split))


if __name__ == '__main__':
    base_dir = '/scratch1/jason/videos/3DPW'
    get_sequences(base_dir, 'val')
