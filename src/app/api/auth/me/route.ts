import { NextRequest, NextResponse } from 'next/server';

// Mock user database - in a real app, this would be a database
const users = [
  {
    id: '1',
    email: 'user@example.com',
    password: 'password123',
    displayName: 'Demo User',
  },
];

export async function GET(request: NextRequest) {
  try {
    // Get token from Authorization header
    const authHeader = request.headers.get('Authorization');
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      );
    }

    const token = authHeader.split(' ')[1];
    
    // In a real app, you would validate the token
    // For this example, we'll decode the token and extract the user ID
    const decoded = Buffer.from(token, 'base64').toString();
    const userId = decoded.split(':')[0];

    // Find user by ID
    const user = users.find(u => u.id === userId);
    
    if (!user) {
      return NextResponse.json(
        { error: 'User not found' },
        { status: 404 }
      );
    }

    // Return user data
    return NextResponse.json({
      id: user.id,
      email: user.email,
      displayName: user.displayName,
    });
  } catch (error) {
    console.error('Auth check error:', error);
    return NextResponse.json(
      { error: 'Authentication failed' },
      { status: 500 }
    );
  }
}