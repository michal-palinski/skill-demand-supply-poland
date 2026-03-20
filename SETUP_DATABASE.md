# PostgreSQL Setup Guide

Your system doesn't have PostgreSQL installed yet. Choose one of the following options:

---

## Option 1: Docker (Easiest - Recommended)

### Prerequisites
- Install Docker Desktop: https://www.docker.com/products/docker-desktop

### Setup Steps

1. **Start PostgreSQL with Docker Compose:**
   ```bash
   docker-compose up -d
   ```

2. **Update your `.env` file:**
   ```env
   # BUR API Credentials
   BUR_USERNAME=your_username
   BUR_API_KEY=your_api_key
   
   # PostgreSQL Database (Docker)
   DB_HOST=localhost
   DB_PORT=5432
   DB_NAME=trainings_pl
   DB_USER=postgres
   DB_PASSWORD=postgres
   ```

3. **Run the scraper:**
   ```bash
   python trainings/scraper_trainings_bur.py
   ```

4. **Stop PostgreSQL when done:**
   ```bash
   docker-compose down
   ```

### Useful Docker Commands

```bash
# View logs
docker-compose logs -f postgres

# Connect to PostgreSQL CLI
docker-compose exec postgres psql -U postgres -d jobs_pl

# Stop database
docker-compose stop

# Start database
docker-compose start

# Remove everything (including data)
docker-compose down -v
```

---

## Option 2: Install PostgreSQL Locally

### Install via Homebrew

```bash
# Install PostgreSQL 16
brew install postgresql@16

# Start PostgreSQL service (runs in background)
brew services start postgresql@16

# Add PostgreSQL to PATH (add this line to ~/.zshrc)
echo 'export PATH="/opt/homebrew/opt/postgresql@16/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### Create Database

```bash
# Create the database
createdb trainings_pl

# Optional: Create a dedicated user
psql trainings_pl -c "CREATE USER jobs_user WITH PASSWORD 'your_password';"
psql trainings_pl -c "GRANT ALL PRIVILEGES ON DATABASE jobs_pl TO jobs_user;"
```

### Update your `.env` file

**Option A: Use your system user (no password needed):**
```env
# BUR API Credentials
BUR_USERNAME=your_username
BUR_API_KEY=your_api_key

# PostgreSQL Database (Local - System User)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trainings_pl
DB_USER=michalpalinski
DB_PASSWORD=
```

**Option B: Use dedicated user:**
```env
# BUR API Credentials
BUR_USERNAME=your_username
BUR_API_KEY=your_api_key

# PostgreSQL Database (Local - Dedicated User)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trainings_pl
DB_USER=jobs_user
DB_PASSWORD=your_password
```

### Run the scraper

```bash
python trainings/scraper_trainings_bur.py
```

---

## Option 3: Use Existing PostgreSQL Server

If you have access to a remote PostgreSQL server, update your `.env`:

```env
# BUR API Credentials
BUR_USERNAME=your_username
BUR_API_KEY=your_api_key

# PostgreSQL Database (Remote)
DB_HOST=your-server.com
DB_PORT=5432
DB_NAME=trainings_pl
DB_USER=your_user
DB_PASSWORD=your_password
```

---

## Verify Connection

Test your database connection:

```bash
python -c "
import asyncio
import asyncpg
import os
from dotenv import load_dotenv
load_dotenv()

async def test():
    try:
        conn = await asyncpg.connect(
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT'),
            database=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD')
        )
        print('✅ Successfully connected to PostgreSQL!')
        await conn.close()
    except Exception as e:
        print(f'❌ Connection failed: {e}')

asyncio.run(test())
"
```

---

## Troubleshooting

### "role does not exist"
- Make sure the user specified in `DB_USER` exists
- Try using your system username: `michalpalinski`

### "database does not exist"
- Create the database: `createdb trainings_pl`
- Or let PostgreSQL auto-create it (if user has privileges)

### "connection refused"
- Make sure PostgreSQL is running
- Docker: `docker-compose ps`
- Local: `brew services list | grep postgresql`

### "password authentication failed"
- Check your `.env` file credentials
- For local installation with system user, try leaving `DB_PASSWORD` empty

