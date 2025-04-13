// Export all services from a single file
import { db } from './database';
import { auth } from './auth';
import { storage } from './storage';

export { db, auth, storage };