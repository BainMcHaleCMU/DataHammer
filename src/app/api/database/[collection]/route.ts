import { NextRequest, NextResponse } from 'next/server';

// Mock database - in a real app, this would be a database
const database: Record<string, Record<string, any>> = {
  users: {
    '1': { id: '1', name: 'John Doe', email: 'john@example.com' },
    '2': { id: '2', name: 'Jane Smith', email: 'jane@example.com' },
  },
  projects: {
    '1': { id: '1', name: 'Project A', owner: '1' },
    '2': { id: '2', name: 'Project B', owner: '2' },
  },
};

export async function GET(
  request: NextRequest,
  { params }: { params: { collection: string } }
) {
  try {
    const { collection } = params;
    const searchParams = request.nextUrl.searchParams;
    
    // Get collection data
    const collectionData = database[collection] || {};
    
    // Convert to array
    let results = Object.values(collectionData);
    
    // Apply filters if provided
    const whereFilters = searchParams.get('where');
    if (whereFilters) {
      try {
        const filters = JSON.parse(whereFilters);
        results = results.filter(item => {
          return filters.every(([field, operator, value]: [string, string, any]) => {
            switch (operator) {
              case '==': return item[field] === value;
              case '!=': return item[field] !== value;
              case '>': return item[field] > value;
              case '>=': return item[field] >= value;
              case '<': return item[field] < value;
              case '<=': return item[field] <= value;
              default: return true;
            }
          });
        });
      } catch (e) {
        console.error('Error parsing where filters:', e);
      }
    }
    
    // Apply ordering if provided
    const orderBy = searchParams.get('orderBy');
    if (orderBy) {
      try {
        const [field, direction] = JSON.parse(orderBy);
        results.sort((a, b) => {
          if (direction === 'desc') {
            return a[field] > b[field] ? -1 : 1;
          }
          return a[field] < b[field] ? -1 : 1;
        });
      } catch (e) {
        console.error('Error parsing orderBy:', e);
      }
    }
    
    // Apply limit if provided
    const limit = searchParams.get('limit');
    if (limit) {
      results = results.slice(0, parseInt(limit, 10));
    }
    
    return NextResponse.json(results);
  } catch (error) {
    console.error(`Error fetching ${params.collection}:`, error);
    return NextResponse.json(
      { error: 'Failed to fetch data' },
      { status: 500 }
    );
  }
}

export async function POST(
  request: NextRequest,
  { params }: { params: { collection: string } }
) {
  try {
    const { collection } = params;
    const data = await request.json();
    
    // Initialize collection if it doesn't exist
    if (!database[collection]) {
      database[collection] = {};
    }
    
    // Generate ID
    const id = Date.now().toString();
    
    // Store document
    const document = { id, ...data };
    database[collection][id] = document;
    
    return NextResponse.json(document);
  } catch (error) {
    console.error(`Error creating document in ${params.collection}:`, error);
    return NextResponse.json(
      { error: 'Failed to create document' },
      { status: 500 }
    );
  }
}