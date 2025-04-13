// Database service to replace Firestore
import axios from 'axios';

interface QueryOptions {
  where?: [string, string, any][];
  orderBy?: [string, 'asc' | 'desc'];
  limit?: number;
  startAfter?: any;
}

interface DatabaseService {
  getDocument: <T>(collection: string, id: string) => Promise<T | null>;
  getDocuments: <T>(collection: string, options?: QueryOptions) => Promise<T[]>;
  addDocument: <T>(collection: string, data: T) => Promise<{ id: string } & T>;
  updateDocument: <T>(collection: string, id: string, data: Partial<T>) => Promise<void>;
  deleteDocument: (collection: string, id: string) => Promise<void>;
}

class LocalDatabaseService implements DatabaseService {
  private apiUrl: string;

  constructor() {
    this.apiUrl = '/api/database';
  }

  async getDocument<T>(collection: string, id: string): Promise<T | null> {
    try {
      const response = await axios.get(`${this.apiUrl}/${collection}/${id}`);
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error) && error.response?.status === 404) {
        return null;
      }
      throw error;
    }
  }

  async getDocuments<T>(collection: string, options?: QueryOptions): Promise<T[]> {
    const response = await axios.get(`${this.apiUrl}/${collection}`, {
      params: options,
    });
    return response.data;
  }

  async addDocument<T>(collection: string, data: T): Promise<{ id: string } & T> {
    const response = await axios.post(`${this.apiUrl}/${collection}`, data);
    return response.data;
  }

  async updateDocument<T>(collection: string, id: string, data: Partial<T>): Promise<void> {
    await axios.patch(`${this.apiUrl}/${collection}/${id}`, data);
  }

  async deleteDocument(collection: string, id: string): Promise<void> {
    await axios.delete(`${this.apiUrl}/${collection}/${id}`);
  }
}

const db = new LocalDatabaseService();

export { db };