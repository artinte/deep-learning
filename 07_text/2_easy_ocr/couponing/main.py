import command


def main():
    assert command.check_adb(), 'Error: the adb was not found'
    command.print_adb_version()
    
    phone_serials = command.get_serials()
    if len(phone_serials) == 0:
        print('No phone connected, the wool-gathering program has exited')
    else:
        print(phone_serials)
    
    for uuid, authorized in phone_serials.items():
        if authorized == 'device':
            command.phone_wake_up(uuid)
            (width, height) = command.phone_size(uuid)
            command.swipe_up(uuid, (width, height))
    
    

if __name__ == '__main__':
    serial_dict = {}
    main()