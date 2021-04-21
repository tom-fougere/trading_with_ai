def sliding_window(nums, period, functions):
    """
    Compute function(s) using a sliding window over a list
    :param nums: list of numbers, digit
    :param period: period length of the window, integer
    :param functions: list of functions to compute, func
    :return:
        - list
    """
    # Initialize list of lists
    result = [[] for i in range(len(functions))]

    # Define the index of the end
    max_index = len(nums)-period+1

    # Loop over the list 'nums'
    for i_index in range(max_index):
        for i_f, f in enumerate(functions):
            result[i_f].append(f(nums[i_index:i_index+period]))

    return result
