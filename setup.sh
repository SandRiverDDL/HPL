# 1. Install Python dependencies
conda create -n HPL python==3.10 -y
conda activate HPL

cd textworld
pip install .
cd ..

pip install -r requirements.txt

cd envs/webshop
python -m spacy download en_core_web_lg
conda install -y -c conda-forge openjdk=11

# 2. Download data for WebShop
# Download data.zip and indexes.zip from WebShop and unzip them in envs/webshop
gdown https://drive.google.com/uc?id=1G_0ccLWn5kZE5rpeyAdh_YuoNzvBUjT9
gdown https://drive.google.com/uc?id=11zOUDkJSgGhYin9NxQtG8PVpDsika86y
unzip data.zip
mkdir search_index
unzip indexes.zip -d search_index/

# 3. Download data for ALFWorld
cd ../../data/alfworld
# Download ALFWorld data and unzip it in data/alfworld
gdown https://drive.google.com/uc?id=1y7Vqeo0_xm9d3I07vZaP6qbPFtyuJ6kI
unzip alfworld_data.zip

# 4. Download data for InterCode-SQL
cd ../intercode_sql
# Download InterCode-SQL data and unzip it in data/intercode_sql
gdown https://drive.google.com/uc?id=19AyZnrniD_NXSbV8mHPh5FgoXjN-5WvP
unzip intercodesql_data.zip

# 5. Create Docker image for InterCode-SQL
cd ../..
docker-compose -f data/intercode_sql/sql-docker-compose.yml up -d

# 6. Download expert trajectories for agent training
# Donwload expert trajectories and unzip them