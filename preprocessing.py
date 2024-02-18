import numpy as np
import pandas as pd 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

def drop_columns(df, columns):
    """
    Drops specified columns from a DataFrame and returns the modified DataFrame.
    Args:
        df (pandas.DataFrame): The DataFrame from which columns need to be dropped.
        columns (list): List of column names (as strings) that should be dropped from the DataFrame.
    Returns:
        pandas.DataFrame: A new DataFrame with the specified columns removed.
    Raises:
        ValueError: If any of the columns specified in the 'columns' list do not exist in the DataFrame.
    """

    if not set(columns).issubset(df.columns):
        raise ValueError(f"One or more columns specified do not exist in the DataFrame. Invalid columns: {set(columns) - set(df.columns)}")
    
    df = df.drop(columns, axis=1)
    return df

def parse_string(matrix_string):
    """
    Convert a MATLAB-style string representation of an array to a NumPy array.
    Args:
        matrix_string (str): A string representing a MATLAB-style matrix, e.g., "[1 2 3; 4 5 6; 7 8 9]".

    Returns:
        numpy.ndarray: A NumPy array representation of the parsed matrix string.

    Raises:
        ValueError: If the provided 'matrix_string' does not adhere to the expected MATLAB-style format.
        TypeError: If the input 'matrix_string' is not a string or if the elements cannot be converted to floats.

    """
    # Ensure the input is a string
    if not isinstance(matrix_string, str):
        raise TypeError("Input must be a string representing a MATLAB-style matrix.")
    
    # Remove the enclosing square brackets and split rows
    rows = matrix_string[1:-1].split(';')
    
    try:
        # Convert string elements to float and create a list of lists
        matrix_list = [list(map(float, row.split())) for row in rows]
    except ValueError as e:
        raise ValueError(f"Failed to convert elements to floats: {e}")
    
    # Convert list of lists to a NumPy array
    array = np.array(matrix_list)
    
    return array

def string_to_row_vector(string):
    """
    Convert a string representation of numbers enclosed in square brackets to a NumPy row vector.
    For tip and top in acpc
    Args:
        string (str): A string containing numbers separated by spaces and enclosed within square brackets, 
                      e.g., "[1.0 2.0 3.0]".
    Returns:
        numpy.ndarray: A NumPy row vector representation of the parsed string, where each element is a float.
    Raises:
        ValueError: If the provided 'string' does not adhere to the expected format or if conversion to float fails.
        TypeError: If the input 'string' is not a string.
    """
    # Ensure the input is a string
    if not isinstance(string, str):
        raise TypeError("Input must be a string containing numbers separated by spaces and enclosed within square brackets.")
    
    try:
        # Strip square brackets, split string at spaces, and convert substrings to floats
        numbers_list = [float(num) for num in string.strip('[]').split()]
    except ValueError as e:
        raise ValueError(f"Failed to convert elements to floats: {e}")
    
    # Convert the list of floats to a NumPy row vector
    row_vector = np.array(numbers_list).reshape(1, -1)
    
    return row_vector

def SDK_transform3d(v3d, T):
    """
    Transform 3D points using a 4x4 transformation matrix.

    Parameters:
    - v3d: 3D points, an array of shape (n, 3)
    - T: Transformation matrix, a 4x4 array

    Returns:
    - transformed: Transformed 3D points, an array of shape (n, 3)
    """
    # Check dimensions of v3d
    if v3d.shape[1] != 3:
        raise ValueError('Wrong dimensions vector3d. Vector should have 3 columns')

    # Check dimensions of T
    if T.shape != (4, 4):
        raise ValueError('Wrong dimensions transformation matrix. T should be 4x4.')

    # Check if T is designed for row multiplication
    if not np.all(np.round(T[0:3, 3], 5) == [0, 0, 0]):
        if np.all(np.round(T[3, 0:3], 5) == [0, 0, 0]):
            # T was probably transposed, automatically repair
            T = T.T
        else:
            raise ValueError('Invalid transformation matrix.')

    # Add 1 to v3d
    v3d_homogeneous = np.hstack((v3d, np.ones((v3d.shape[0], 1))))

    # Perform transformation
    transformed_homogeneous = np.dot(v3d_homogeneous, T)

    # Clip off extra column
    transformed = transformed_homogeneous[:, :3]

    return transformed

def contact_coordinates(points, leadtype, transformation_matrix):
    """convert active and grounded contacts to their respective coordinate spaces

    Args:
        points (str): Midpoint of activated electrode (derived by me using some weird logic)
        leadtype (str): Medtronic 3389 or 3387
        transformation_matrix (str): transformation matrix for LeadDBS sapce to another brain atlas

    Raises:
        ValueError: _description_
        ValueError: _description_

    Returns:
        _type_: the coordinates of the active point in desired space
    """
    contacts = [float(num) for num in points.split(' ')]
    contact_point = [0, 0, 0]
    distance = {'Medtronic3389': 2, 'Medtronic3387': 3}.get(leadtype, None)
    if contacts.count(1) == 1:   
        index = contacts.index(1)     
        contact_point = [0,0, 2.25 + (index * distance)]
    elif contacts.count(1) == 2:
        indices_of_ones = [i for i, value in enumerate(contacts) if value == 1]
        if indices_of_ones[1] - indices_of_ones[0] == 1:
            midpoint_index = (indices_of_ones[0] + indices_of_ones[1]) / 2
            contact_point = [0,0, 2.25 + (midpoint_index * distance)]          
        else:
            raise ValueError("The '1' values are not consecutive.")
    else:
        raise ValueError("The list does not have exactly two '1' values.")
    pole_coordinates = np.array(contact_point) 
    transform = parse_string(transformation_matrix)
    pole_coor = pole_coordinates.reshape(1,-1)  
    pole = SDK_transform3d(pole_coor, transform)
    x,y,z = pole[0][0],pole[0][1],pole[0][2] 
    return x,y,z


def electode_extremes(transformation_matrix):
    point = [0,0,0]
    pole_coordinates = np.array(point) 
    transform = parse_string(transformation_matrix)
    pole_coor = pole_coordinates.reshape(1,-1)  
    pole = SDK_transform3d(pole_coor, transform)
    x,y,z =pole[0][0],pole[0][1],pole[0][2] 
    return x,y,z

def remove_totals(df, target):
    """There are different metrics for which the improvement score was calculated.
        The user should be able to select the metric they want to use

    Args:
        df (DataFrame): _description_
        target (float): improvement score to be used

    Returns:
        DataFrame: _description_
    """
    # Get the list of all total columns
    all_totals = [col for col in df.columns if 'total' in col]

    # Remove the selected total column from the list
    all_totals.remove(target)

    # Drop the unwanted total columns from the DataFrame
    df = df.drop(all_totals, axis=1)

    return df

def preprocess_data(df, to_delete, space, target):
    
    # if not to_delete:
    #     pass
    # else:
    #     non_existing_columns = set(to_delete) - set(df.columns)
    #     if non_existing_columns:
    #         raise ValueError(f"One or more columns specified in 'to_delete' do not exist in the DataFrame: {non_existing_columns}")
    #     else:
    #         df1 = df.drop(to_delete, axis=1)

    df1 = remove_totals(df, target)

    if space == 'MNI':
        transformation_matrix = 'Tlead2MNI'
    elif space == 'ACPC':
        transformation_matrix = 'Tlead2ACPC'
    else:
        print("Invalid space specified. Please choose either MNI or ACPC.")

    df1[['electrode tip (x)', 'electrode tip (y)', 'electrode tip (z)']] = df1.apply(lambda row: electode_extremes(row[transformation_matrix]), axis=1).apply(pd.Series)
    df1[['active contact (x)', 'active contact (y)', 'active contact (z)']] = df1.apply(lambda row: contact_coordinates(row['position'], row['leadtype'], row[transformation_matrix]), axis=1).apply(pd.Series)

    drop_please = ['activecontact', 'groundedcontact', 'position', 'tipinacpc', 'topinacpc',
                   'Tlead2MNI', 'Tlead2ACPC', 'leadtype']
    df1 = df1.drop(drop_please, axis=1)

    label = df1.pop(target)
    df1['total'] = label

    return df1

def normalise_data(df, to_delete, space, target):

    df = preprocess_data(df, to_delete, space, target)
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]
    try:
        ct = make_column_transformer(
            (StandardScaler(), make_column_selector(dtype_include=np.number)),
            (OrdinalEncoder(categories='auto'), make_column_selector(dtype_include=object)))
        X = ct.fit_transform(X)
        
    except Exception as e:
        print(f"Error in normalising data: {e}")
        return None
    else:
        return {
        "preprocessor": ct,
        "data": (X,y)
        }  

def inference_preprocessor(data_file, target, to_delete, space, ct):
    data = pd.read_csv(data_file)
    data = preprocess_data(data, to_delete, space, target)
    data = ct.transform(data)
    return data



