using System;

namespace MatrixLibrary
{
	public class MatrixException : Exception
    {
        public MatrixException(string message)
            : base(message) { }
    }

    public class Matrix : ICloneable
    {
        public int Rows { get; }

        public int Columns { get; }

        public double[,] Array { get; }

        public Matrix(int rows, int columns)
        {
            if (rows < 0 || columns < 0)
			{
                throw new ArgumentOutOfRangeException("rows",
                    "Rows should be more than 0");
            }
            else if(columns < 0)
			{
                throw new ArgumentOutOfRangeException("columns",
                    "Columns should be more than 0");
            }

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

        public object Clone() => new Matrix(Array);

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

			if (matrix1.Rows != matrix2.Rows || 
                matrix1.Columns != matrix2.Columns)
			{
                throw new MatrixException("Matrixes should have same dimensions");
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

            if (matrix1.Rows != matrix2.Rows ||
                matrix1.Columns != matrix2.Columns)
            {
                throw new MatrixException("Matrixes should have same dimensions");
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

            if (matrix1.Columns != matrix2.Rows)
            {
                throw new MatrixException("Matrixes should have same dimensions");
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

        public Matrix Add(Matrix matrix)
        {
            if (matrix == null)
            {
                throw new ArgumentNullException("matrix",
                    "matrix shouldn't be null");
            }

            if (Rows != matrix.Rows ||
                Columns != matrix.Columns)
            {
                throw new MatrixException("Matrixes should have same dimensions");
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

        public Matrix Subtract(Matrix matrix)
        {
            if (matrix == null)
            {
                throw new ArgumentNullException("matrix",
                    "matrix shouldn't be null");
            }

            if (Rows != matrix.Rows ||
                Columns != matrix.Columns)
            {
                throw new MatrixException("Matrixes should have same dimensions");
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

        public Matrix Multiply(Matrix matrix)
        {
            if (matrix == null)
            {
                throw new ArgumentNullException("matrix",
                    "matrix shouldn't be null");
            }

            if (Columns != matrix.Rows)
            {
                throw new MatrixException("Matrixes should have same dimensions");
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

        public override bool Equals(object obj)
        {
            bool areEqual = false;

            if (obj is Matrix matrix && 
                Rows == matrix.Rows && Columns == matrix.Columns)
            {
				for (int i = 0; i < matrix.Rows; i++)
				{
					for (int j = 0; j < matrix.Columns; j++)
					{
						if (Array[i, j] == matrix[i, j])
						{
                            areEqual = true;
                        }
					}
        		}
            }

            return areEqual;
        }

		public override int GetHashCode()
		{
			return base.GetHashCode();
		}
	}
}
