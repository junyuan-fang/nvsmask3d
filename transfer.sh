#!/bin/bash

ids=(
'7b6477cb95' 'c50d2d1d42' 'cc5237fd77' 'acd95847c5' 'fb5a96b1a2' 'a24f64f7fb'
'1ada7a0617' '5eb31827b7' '3e8bba0176' '3f15a9266d' '21d970d8de' '5748ce6f01'
'c4c04e6d6c' '7831862f02' 'bde1e479ad' '38d58a7a31' '5ee7c22ba0' 'f9f95681fd'
'3864514494' '40aec5fffa' '13c3e046d7' 'e398684d27' 'a8bf42d646' '45b0dac5e3'
'31a2c91c43' 'e7af285f7d' '286b55a2bf' '7bc286c1b6' 'f3685d06a9' 'b0a08200c9'
'825d228aec' 'a980334473' 'f2dc06b1d2' '5942004064' '25f3b7a318' 'bcd2436daf'
'f3d64c30f8' '0d2ee665be' '3db0a1c8f3' 'ac48a9b736' 'c5439f4607' '578511c8a9'
'd755b3d9d8' '99fa5c25e1' '09c1414f1b' '5f99900f09' '9071e139d9' '6115eddb86'
'27dd4da69e' 'c49a8c6cff'
)
# Define local and remote directories
local_dir="/home/fangj1/Code/nerfstudio-nvsmask3d/nvsmask3d/data/ScannetPP/data"
remote_user="fangjuny"
remote_host="puhti.csc.fi"
remote_dir="/scratch/project_2003267/junyuan/data"

# Loop through each ID and rsync matching folders/files
for id in "${ids[@]}"; do
  # Ensure the folder and all subdirectories/files are completely re-synced, excluding 'iphone' and 'render_rgb' subdirectories
  rsync -avzc --progress --info=progress2  --exclude 'iphone/' --exclude 'dslr/render_rgb/' --delete "$local_dir/$id/" "$remote_user@$remote_host:$remote_dir/$id/"
done

# # Delete folders on remote that are not in the ID list
# ssh "$remote_user@$remote_host" bash << EOF
#   cd "$remote_dir"
#   for dir in */; do
#     dir_id=\${dir%/}
#     if [[ ! " ${ids[*]} " =~ " \$dir_id " ]]; then
#       echo "Deleting remote directory: \$dir_id"
#       rm -rf "\$dir_id"
#     fi
#   done
# EOF

# echo "Sync complete!"
