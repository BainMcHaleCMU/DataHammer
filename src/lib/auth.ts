// Authentication service to replace Firebase Auth
import axios from 'axios';

export interface User {
  id: string;
  email: string;
  displayName?: string;
  photoURL?: string;
}

export interface AuthService {
  currentUser: User | null;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<User>;
  register: (email: string, password: string, displayName?: string) => Promise<User>;
  logout: () => Promise<void>;
  getToken: () => Promise<string | null>;
  onAuthStateChanged: (callback: (user: User | null) => void) => () => void;
}

class LocalAuthService implements AuthService {
  private apiUrl: string;
  private _currentUser: User | null = null;
  private listeners: ((user: User | null) => void)[] = [];

  constructor() {
    this.apiUrl = '/api/auth';
    // Check if user is already logged in (e.g., from localStorage)
    this.checkAuthState();
  }

  private async checkAuthState() {
    const token = localStorage.getItem('auth_token');
    if (token) {
      try {
        const response = await axios.get(`${this.apiUrl}/me`, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });
        this._currentUser = response.data;
        this.notifyListeners();
      } catch (error) {
        localStorage.removeItem('auth_token');
        this._currentUser = null;
      }
    }
  }

  private notifyListeners() {
    this.listeners.forEach(listener => listener(this._currentUser));
  }

  get currentUser(): User | null {
    return this._currentUser;
  }

  get isAuthenticated(): boolean {
    return !!this._currentUser;
  }

  async login(email: string, password: string): Promise<User> {
    const response = await axios.post(`${this.apiUrl}/login`, {
      email,
      password,
    });

    const { user, token } = response.data;
    localStorage.setItem('auth_token', token);
    this._currentUser = user;
    this.notifyListeners();
    return user;
  }

  async register(email: string, password: string, displayName?: string): Promise<User> {
    const response = await axios.post(`${this.apiUrl}/register`, {
      email,
      password,
      displayName,
    });

    const { user, token } = response.data;
    localStorage.setItem('auth_token', token);
    this._currentUser = user;
    this.notifyListeners();
    return user;
  }

  async logout(): Promise<void> {
    await axios.post(`${this.apiUrl}/logout`);
    localStorage.removeItem('auth_token');
    this._currentUser = null;
    this.notifyListeners();
  }

  async getToken(): Promise<string | null> {
    return localStorage.getItem('auth_token');
  }

  onAuthStateChanged(callback: (user: User | null) => void): () => void {
    this.listeners.push(callback);
    // Initial call with current state
    callback(this._currentUser);
    
    // Return unsubscribe function
    return () => {
      this.listeners = this.listeners.filter(listener => listener !== callback);
    };
  }
}

const auth = new LocalAuthService();

export { auth };