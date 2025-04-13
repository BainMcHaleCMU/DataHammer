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
  { params }: { params: { collection: string; id: string } }
) {
  try {
    const { collection, id } = params;
    
    // Check if collection exists
    if (!database[collection]) {
      return NextResponse.json(
        { error: 'Collection not found' },
        { status: 404 }
      );
    }
    
    // Check if document exists
    if (!database[collection][id]) {
      return NextResponse.json(
        { error: 'Document not found' },
        { status: 404 }
      );
    }
    
    return NextResponse.json(database[collection][id]);
  } catch (error) {
    console.error(`Error fetching document ${params.id}:`, error);
    return NextResponse.json(
      { error: 'Failed to fetch document' },
      { status: 500 }
    );
  }
}

export async function PATCH(
  request: NextRequest,
  { params }: { params: { collection: string; id: string } }
) {
  try {
    const { collection, id } = params;
    const data = await request.json();
    
    // Check if collection exists
    if (!database[collection]) {
      return NextResponse.json(
        { error: 'Collection not found' },
        { status: 404 }
      );
    }
    
    // Check if document exists
    if (!database[collection][id]) {
      return NextResponse.json(
        { error: 'Document not found' },
        { status: 404 }
      );
    }
    
    // Update document
    database[collection][id] = {
      ...database[collection][id],
      ...data,
    };
    
    return NextResponse.json(database[collection][id]);
  } catch (error) {
    console.error(`Error updating document ${params.id}:`, error);
    return NextResponse.json(
      { error: 'Failed to update document' },
      { status: 500 }
    );
  }
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: { collection: string; id: string } }
) {
  try {
    const { collection, id } = params;
    
    // Check if collection exists
    if (!database[collection]) {
      return NextResponse.json(
        { error: 'Collection not found' },
        { status: 404 }
      );
    }
    
    // Check if document exists
    if (!database[collection][id]) {
      return NextResponse.json(
        { error: 'Document not found' },
        { status: 404 }
      );
    }
    
    // Delete document
    delete database[collection][id];
    
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error(`Error deleting document ${params.id}:`, error);
    return NextResponse.json(
      { error: 'Failed to delete document' },
      { status: 500 }
    );
  }
}