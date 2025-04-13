/**
 * Service for data processing and report generation
 * Connects to the backend API
 */

export interface ProcessingResult {
  success: boolean;
  message: string;
  data?: any;
  error?: string;
}

// Base URL for API calls
const API_BASE_URL = 'http://localhost:12001';

/**
 * Process data based on user instructions
 * @param instructions User-provided instructions for data processing
 * @param dataType Type of data to process (csv, json, xml, text)
 * @param file The file to process
 * @returns Promise with processing result
 */
export const processData = async (
  instructions: string,
  dataType: string,
  file?: File
): Promise<ProcessingResult> => {
  console.log(`Processing ${dataType} data with instructions: ${instructions}`);
  
  try {
    // If no file is provided, return a mock response
    if (!file) {
      console.warn('No file provided, returning mock response');
      return {
        success: true,
        message: 'Data processed successfully (mock)',
        data: {
          summary: 'Processed data according to instructions (mock)',
          dataType,
          instructionsApplied: instructions,
          timestamp: new Date().toISOString(),
        },
      };
    }
    
    // Create form data for the API request
    const formData = new FormData();
    formData.append('file', file);
    
    // Parse instructions into goals
    const goals = instructions.split('\n')
      .filter(line => line.trim().length > 0)
      .map(line => line.trim());
    
    // Add goals to form data
    formData.append('goals', JSON.stringify(goals));
    
    // Make API request to the reporting endpoint
    const response = await fetch(`${API_BASE_URL}/agent-swarm/report`, {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.detail || 'Failed to process data');
    }
    
    const responseData = await response.json();
    
    return {
      success: true,
      message: 'Data processed successfully',
      data: responseData.report,
    };
  } catch (error) {
    console.error('Error processing data:', error);
    return {
      success: false,
      message: 'Failed to process data',
      error: error instanceof Error ? error.message : 'Unknown error',
    };
  }
};

/**
 * Validate user instructions
 * @param instructions User-provided instructions
 * @returns Object with validation result
 */
export const validateInstructions = (instructions: string): { 
  valid: boolean; 
  error?: string;
} => {
  if (!instructions || instructions.trim() === '') {
    return {
      valid: false,
      error: 'Instructions cannot be empty',
    };
  }
  
  if (instructions.length < 10) {
    return {
      valid: false,
      error: 'Instructions must be at least 10 characters long',
    };
  }
  
  return { valid: true };
};