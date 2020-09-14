import shutil
import fileinput


def dict_replace(template_file, output_file, dictionary):        
    # Read in the template file
    with open(template_file) as f:
        data = f.read()
        
    # Make substitutions in the template file text
    for k,v in dictionary.items():
        data = data.replace("$"+str(k), str(v))
    
    # Save the modified template as a new file
    with open(output_file, 'w') as writer:
        writer.write(data)
    return output_file