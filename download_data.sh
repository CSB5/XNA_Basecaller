#!/bin/bash
set -e ## For exiting on first error encountered

download_and_extract_gdrive() {
    local file_id="$1"  # Google Drive file ID
    local base_dir="$2" # Base folder to extract the tarball
    local dest_dir="$3" # Destination folder resulting from tarball
    local tar_filename="$4" # tarball filename
    
	local full_dest_dir="$base_dir/$dest_dir"

    if [ -e "$full_dest_dir" ]; then # Check if the destination folder already exists
        echo "> '$full_dest_dir' found. Skipping download and extraction."
        return 0
    fi

    local temp_tar="/tmp/$tar_filename"

    # echo "Downloading file from Google Drive (ID: $file_id)..."
    echo "Downloading file $temp_tar ..."
    curl -L -o "$temp_tar" "https://drive.usercontent.google.com/download?id=$file_id&confirm=z" || {
        echo "Failed to download the file from Google Drive."
        return 1
    }

	if [[ ! "$dest_dir" == "$tar_filename" ]]; then
		echo "Extracting tarball to '$base_dir'..."
		# Create the destination folder and extract the tarball
		mkdir -p "$base_dir"
		if [[ "$temp_tar" == *.gz ]]; then
			tar -xzf "$temp_tar" -C "$base_dir" || {
				echo "Failed to extract the gzipped tarball."
				rm -f "$temp_tar" # Clean up the temporary file
				return 1
			}
		else
			tar -xf "$temp_tar" -C "$base_dir" || {
				echo "Failed to extract the tarball."
				rm -f "$temp_tar" # Clean up the temporary file
				return 1
			}
		fi
		echo "Extraction completed."
	else # Extracting not necessary, just move tar file to base directory
		mv -v "$temp_tar" "$base_dir"
    fi
	
    rm -f "$temp_tar"
}

echo "> Downloading data started" - `date`
echo "Estimated total size required for all downloads: 11GB" # 0.1+1.78+1.31+3.15+4.64
printf "\n"

ls ub-bonito/ xna_libs/ > /dev/null # sanity-check to validate it is in the correct folder

printf   "+++++ XNA baseline model ++++++++++++++++++++++++++++++++++++++++++\n"
download_and_extract_gdrive '1tzkkEDFFe_cCsWO2m_XRg04RMePoeIVA' \
	'ub-bonito/bonito/models/xna_r9.4.1_e8_sup@v3.3' 'weights_1.tar' 'weights_1.tar'

printf "\n+++++ POC Eval reads ++++++++++++++++++++++++++++++++++++++++++++++\n"
download_and_extract_gdrive '1mFjSgkIEDSjh-2Z7_UdQQU3pUIw9XMCN' \
	'xna_libs/POC' 'reads' 'POC-reads.tar'

printf "\n+++++ Complex Library Eval reads ++++++++++++++++++++++++++++++++++\n"
download_and_extract_gdrive '1ulyUDKnp6b43x3Ms-pwD584VpamzSY-S' \
	'xna_libs/CPLX' 'reads' 'CPLX-reads.tar.gz'

printf "\n+++++ XNA train data ++++++++++++++++++++++++++++++++++++++++++++++\n"
download_and_extract_gdrive '1RSVfTaCSv1QTGf-sUmpQmBg5ne6CBCy-' \
	'ub-bonito/bonito/data' 'xna_r9.4.1-sampled' 'xna_r9.4.1-sampled.tar'
download_and_extract_gdrive '10z9J_itvw6CAiB6GSUbDVjIKd4Hwq5UY' \
	'ub-bonito/bonito/data' 'xna_r9.4.1' 'xna_r9.4.1.tar'

printf "\n+++++ DNA train data ++++++++++++++++++++++++++++++++++++++++++++++\n"
download_and_extract_gdrive '1x3y3j6ru2PUIEGDoXJnNsYdnw3p-T5zq' \
	'ub-bonito/bonito/data/dna_r9.4.1' 'sampled_0.01' 'sampled_0.01.tar'
download_and_extract_gdrive '1I8vfBeh2ZlM6uUjlxJdNvZbcyIv2kxWV' \
	'ub-bonito/bonito/data/dna_r9.4.1' 'sampled_0.25' 'sampled_0.25.tar'

printf "\n"
echo "> Downloading data finished " - `date`
