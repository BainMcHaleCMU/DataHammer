# DataHammer: Advanced Data Analytics Platform

A completely revamped Next.js application with Chakra UI and TypeScript for data analytics. This application allows users to upload spreadsheet files for data analysis, cleaning, visualization, and predictive modeling.

> **Note**: This PR proposes a complete overhaul of the DataHammer repository with a modern tech stack and enhanced features.

## Features

- File upload interface for spreadsheet data (CSV, XLS, XLSX)
- Data cleaning and preprocessing
- Exploratory data analysis
- Data visualization
- Insights generation
- Predictive modeling
- Firebase integration for data storage
- GitHub Pages deployment

## Tech Stack

- **Frontend**: Next.js, TypeScript, Chakra UI, React Dropzone
- **Backend**: FastAPI, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Database**: Firebase Firestore
- **Deployment**: GitHub Pages

## Getting Started

### Prerequisites

- Node.js (v18 or higher)
- Python (v3.8 or higher)
- pip (Python package manager)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/BainMcHale/DataHammer.git
   cd DataHammer
   ```

2. Install frontend dependencies:
   ```bash
   npm install
   ```

3. Install backend dependencies:
   ```bash
   cd backend
   pip install fastapi uvicorn pandas numpy scikit-learn matplotlib seaborn
   ```

### Configuration

1. Create a Firebase project and add your configuration to `.env.local`:
   ```
   NEXT_PUBLIC_FIREBASE_API_KEY=your-api-key
   NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN=your-auth-domain
   NEXT_PUBLIC_FIREBASE_PROJECT_ID=your-project-id
   NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET=your-storage-bucket
   NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID=your-messaging-sender-id
   NEXT_PUBLIC_FIREBASE_APP_ID=your-app-id
   NEXT_PUBLIC_FIREBASE_MEASUREMENT_ID=your-measurement-id
   ```

### Running the Application

1. Start the backend server:
   ```bash
   cd backend
   python main.py
   ```

2. In a separate terminal, start the frontend:
   ```bash
   npm run dev
   ```

3. Or use the provided script to run both:
   ```bash
   ./run.sh
   ```

4. Open your browser and navigate to `http://localhost:3000`

## Deployment

The application is configured for GitHub Pages deployment using GitHub Actions. When you push to the main branch, the workflow will automatically build and deploy the application.

To configure GitHub Pages:

1. Go to your repository settings
2. Navigate to Pages
3. Select the `gh-pages` branch as the source
4. Add your Firebase secrets to the repository secrets
