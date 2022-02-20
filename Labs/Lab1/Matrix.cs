using System;

namespace MatrixLibrary
{
    public class Matrix : ICloneable
    {
        public int Rows { get; }

        public int Columns { get; }
        
        public double[,] Array { get; }
        
        /// <exception cref="ArgumentOutOfRangeException"></exception>
        public Matrix(int rows, int columns)
        {
            Rows = rows;
            Columns = columns;

            Array = new double[rows, columns];
        }

        public Matrix(double[,] array)
        {
			Array = array ?? throw new ArgumentNullException("array");
            Rows = array.GetLength(0);
            Columns = array.GetLength(1);
        }

        public double this[int row, int column]
        {
            get 
            {
                if (row < 0 || column < 0)
                {
                    throw new ArgumentException("indexes can't be less than 0");
                }
                
                return Array[row, column]; }
            set 
            {
                if (row < 0 || column < 0)
				{
                    throw new ArgumentException("indexes can't be less than 0");
                }

                Array[row, column] = value; 
            }
        }

        public object Clone()
        {
            Matrix deepClone = new Matrix(Rows, Columns);

            return deepClone;
        }

        /// <exception cref="MatrixException"></exception>
        public static Matrix operator +(Matrix matrix1, Matrix matrix2)
        {
            if (matrix1 == null)
			{
                throw new ArgumentNullException("matrix1",
                    "matrix1 shouldn't be null");
			}
            else if (matrix2 == null)
			{
                throw new ArgumentNullException("matrix2",
                    "matrix2 shouldn't be null");
            }

            Matrix result = new Matrix(matrix1.Rows, matrix1.Columns);

			for (int i = 0; i < result.Rows; i++)
			{
				for (int j = 0; j < result.Columns; j++)
				{
                    result[i, j] = matrix1[i, j] + matrix2[i, j];
                }
			}

            return result;
        }

        /// <exception cref="MatrixException"></exception>
        public static Matrix operator -(Matrix matrix1, Matrix matrix2)
        {
            if (matrix1 == null)
            {
                throw new ArgumentNullException("matrix1",
                    "matrix1 shouldn't be null");
            }
            else if (matrix2 == null)
            {
                throw new ArgumentNullException("matrix2",
                    "matrix2 shouldn't be null");
            }

            Matrix result = new Matrix(matrix1.Rows, matrix1.Columns);

            for (int i = 0; i < result.Rows; i++)
            {
                for (int j = 0; j < result.Columns; j++)
                {
                    result[i, j] = matrix1[i, j] - matrix2[i, j];
                }
            }

            return result;
        }

        /// <exception cref="MatrixException"></exception>
        public static Matrix operator *(Matrix matrix1, Matrix matrix2)
        {
            if (matrix1 == null)
            {
                throw new ArgumentNullException("matrix1",
                    "matrix1 shouldn't be null");
            }
            else if (matrix2 == null)
            {
                throw new ArgumentNullException("matrix2",
                    "matrix2 shouldn't be null");
            }

            Matrix result = new Matrix(matrix1.Rows, matrix2.Columns);

            for (int i = 0; i < result.Rows; i++)
            {
                for (int j = 0; j < result.Columns; j++)
                {
					for (int k = 0; k < matrix2.Rows; k++)
					{
                        result[i, j] += matrix1[i,k] * matrix2[k,j];
                    }
                }
            }

            return result;
        }

        /// <exception cref="MatrixException"></exception>
        public Matrix Add(Matrix matrix)
        {
            if (matrix == null)
            {
                throw new ArgumentNullException("matrix",
                    "matrix shouldn't be null");
            }

            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
                {
                    Array[i, j] = Array[i, j] + matrix[i, j];
                }
            }

            return this;
        }

        /// <exception cref="MatrixException"></exception>
        public Matrix Subtract(Matrix matrix)
        {
            if (matrix == null)
            {
                throw new ArgumentNullException("matrix",
                    "matrix shouldn't be null");
            }

            for (int i = 0; i < Rows; i++)
            {
                for (int j = 0; j < Columns; j++)
                {
                    Array[i, j] = Array[i, j] - matrix[i, j];
                }
            }

            return this;
        }

        /// <exception cref="MatrixException"></exception>
        public Matrix Multiply(Matrix matrix)
        {
            if (matrix == null)
            {
                throw new ArgumentNullException("matrix",
                    "matrix shouldn't be null");
            }

            Matrix result = new Matrix(Rows, matrix.Columns);

            for (int i = 0; i < result.Rows; i++)
            {
                for (int j = 0; j < result.Columns; j++)
                {
                    for (int k = 0; k < matrix.Rows; k++)
                    {
                        result[i, j] += Array[i, k] * matrix[k, j];
                    }
                }
            }

            return result;
        }

        /// <summary>
        /// Tests if <see cref="Matrix"/> is identical to this Matrix.
        /// </summary>
        /// <param name="obj">Object to compare with. (Can be null)</param>
        /// <returns>True if matrices are equal, false if are not equal.</returns>
        public override bool Equals(object obj)
        {
            throw new NotImplementedException();
        }

		public override int GetHashCode()
		{
			return base.GetHashCode();
		}
	}
}
