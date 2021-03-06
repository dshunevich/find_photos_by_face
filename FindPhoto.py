import dlib
import os
from skimage import io
from scipy.spatial import distance
import shutil

dir_path = 'photos/'
result_path = 'finded_photos/'
example_photo = 'find.jpg'
quantity = 0
files_action = 'copy' #  copy, move

# get list of photos from folder 'photos'
def getfilelist(dir_path):
    global quantity
    mas = []
    for root, dirs, files in os.walk(dir_path):
        for name in files:
            fullname = os.path.join(root, name)
            if('.jpg' in fullname or '.JPEG' in fullname or '.JPG' in fullname or '.jpeg' in fullname):
                mas.append(fullname)
    quantity = str(len(mas))
    print('Готовится к анализу: '+quantity+' фотографий')
    return mas

# get a digital representation of the faces found on the photo
# function returns an array with face biometrics
def get_face_descriptors(filename):
    faces_list = []
    img = io.imread(filename)
    detected_faces = detector(img, 1)
    shape = None
    face_descriptor = None
    for k, d in enumerate(detected_faces):
        shape = sp(img, d)
        try:
            face_descriptor = face_rec.compute_face_descriptor(img, shape)
            if (face_descriptor != None):
                faces_list.append(face_descriptor)
        except Exception as ex:
            pass
    return faces_list

# Run finding faces for example photo and copy finded photos in folder "result_path"
if __name__ == '__main__':
    sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    face_rec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
    detector = dlib.get_frontal_face_detector()
    min_distance = 2

    f1 = get_face_descriptors(example_photo)[0]
    files = getfilelist(dir_path)
    flag = 0
    for f in files:
        file_name, file_extension = os.path.splitext(f)
        print(file_name, file_extension)
        flag = flag+1
        print('Анализ ' +f+' - '+str(flag)+' фото из '+str(quantity))
        if os.path.exists(f):
            try:
                find_faces = get_face_descriptors(f)
                print('На фото: ' + str(len(find_faces)) + ' лиц')
                for f2 in find_faces:
                    if f2 != []:
                        euc_distance = distance.euclidean(f1, f2)
                        print(euc_distance)
                        if euc_distance < 0.65:
                            print('Найдено лицо: '+f)
                            if files_action == 'move':
                                shutil.move(f, result_path+str(flag)+file_extension)
                            elif files_action == 'copy':
                                shutil.copyfile(f, result_path + str(flag) + file_extension)
            except:
                continue