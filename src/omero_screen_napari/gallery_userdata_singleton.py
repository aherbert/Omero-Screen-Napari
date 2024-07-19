from omero_screen_napari.gallery_userdata import UserData

userdata = UserData()

def reset_userdata():
    global userdata
    userdata.reset()  # Reset the data class
# def initialize_userdata(user_data_dict=None):
#     print(f"inside function: {omero_data.channel_data.keys()}")
#     if user_data_dict:
#         UserData.set_omero_data_channel_keys(user_data_dict["_omero_data_channel_keys"])
#         return UserData(**user_data_dict)
#     if channel_keys := list(omero_data.channel_data.keys()):
#         UserData.set_omero_data_channel_keys(channel_keys)
#         return UserData.set_defaults(channel_keys)
#     else:
#         return None

# userdata = initialize_userdata()

