import { useEffect, useMemo, useRef, useState } from 'react';
import { useSelector } from 'react-redux';
import { useLocation, useNavigate } from 'react-router-dom';

import PanelTargetGraph from '@/components/ui/panel-target-graph/PanelTargetGraph';

import { useActions } from '@/hooks/useActions';

import {
	useGetUserFoldersQuery,
	useGetUserIdQuery,
} from '../../../services/other.service';
import ProgressBar from '../../ui/progress-bar/ProgressBar';

import styles from './DataSet.module.scss';
import Folder from './folder/Folder';
import HistoryCard from './history-card/HistoryCard';
import MosinformArchive from './MosinformArchive';
import MlopsQueue from './MlopsQueue';
import NoData from './no-data/NoData';
import { dataSetButtons } from '@/data/panel.data';

const MOSINFORM_TAB = 'Мосинформ.Рейтинг';
const QUEUE_TAB = 'Очередь ML';

const DataSet = () => {
	const { addButtonTarget_PopupDelete } = useActions();
	const tabOneRef = useRef(null);
	const tabTwoRef = useRef(null);
	const tabThreeRef = useRef(null);
	const [elementWidth, setElementWidth] = useState({
		one: 0,
		two: 0,
		three: 0,
	});
	const location = useLocation();
	const navigate = useNavigate();
	const [activeButton, setActiveButton] = useState(() => {
		const tab = new URLSearchParams(location.search).get('tab');
		if (tab === 'mosinform') return MOSINFORM_TAB;
		if (tab === 'queue') return QUEUE_TAB;
		return 'Файлы данных';
	});
	const [showDevModal, setShowDevModal] = useState(false);

	const {
		data: data_getUserId,
		isError: isError_getUserId,
		error: error_getUserId,
		isLoading: isLoading_getUserId,
	} = useGetUserIdQuery();
	const { data, isError, error, isLoading, isSuccess } =
		useGetUserFoldersQuery(data_getUserId);

	const { processedData } = useSelector(state => state.folderTarget);
	const { index_doc, post, progress_load } = useSelector(state => state.aiData);
	const dataForRequest = useSelector(state => state.dataForRequest);

	useEffect(() => {
		if (activeButton === 'Файлы данных') {
			addButtonTarget_PopupDelete(activeButton);
		} else {
			addButtonTarget_PopupDelete(activeButton);
		}
	}, [activeButton]);

	useEffect(() => {
		if (tabOneRef.current) {
			const width = tabOneRef.current.offsetWidth;
			setElementWidth(prev => ({
				...prev,
				one: width,
			}));
		}
		if (tabTwoRef.current) {
			const width = tabTwoRef.current.offsetWidth;
			setElementWidth(prev => ({
				...prev,
				two: width,
			}));
		}
		if (tabThreeRef.current) {
			const width = tabThreeRef.current.offsetWidth;
			setElementWidth(prev => ({
				...prev,
				three: width,
			}));
		}
	}, []);

	useEffect(() => {
		const tab = new URLSearchParams(location.search).get('tab');
		if (tab === 'mosinform') setActiveButton(MOSINFORM_TAB);
		if (tab === 'queue') setActiveButton(QUEUE_TAB);
	}, [location.search]);

	const onClick = button => {
		if (button === 'Файлы кластеризации авторов') {
			setShowDevModal(true);
			return;
		}
		setActiveButton(button);
		if (button === MOSINFORM_TAB) {
			navigate('/data-set?tab=mosinform', { replace: true });
		} else if (button === QUEUE_TAB) {
			navigate('/data-set?tab=queue', { replace: true });
		} else if (location.search.includes('tab=')) {
			navigate('/data-set', { replace: true });
		}
	};

	const {
		json_files_directory: dataUser,
		projector_files_directory: dataUser_Projector,
		bertopic_files_directory: dataUser_bertopic,
	} = useSelector(store => store.dataUsersSlice);

	const allData = Object.keys(
		activeButton === 'Файлы данных' ? dataUser : dataUser_Projector || {},
	);
	const arrayData =
		dataUser_bertopic && Object.keys(dataUser_bertopic).length > 0
			? dataUser_bertopic
			: {};

	const file_name = Object.values(dataUser)
		.flat()
		.find(file => dataForRequest.index === file.index_number);

	const [filterText, setFilterText] = useState('');
	const getFilteredData = (data, filterText) => {
		return (
			data &&
			data.filter(folder =>
				folder.toLowerCase().includes(filterText.toLowerCase()),
			)
		);
	};

	const filteredData = useMemo(() => {
		if (activeButton === 'Файлы данных') {
			return getFilteredData(allData, filterText);
		} else {
			return [];
		}
	}, [activeButton, allData, processedData, filterText]);

	const handleInputChange = event => {
		setFilterText(event.target.value);
	};

	console.log('file_name', file_name, index_doc);

	const renderContent = (activeButton, allData) => {
		if (activeButton === 'Файлы данных') {
			return (
				<>
					{allData && allData.values && allData.values.length !== 0 ? (
						allData.values.map(folder => (
							<Folder
								key={Math.random() + Math.random()}
								folder={folder}
								processedFolder={false}
								buttonTarget={activeButton}
							/>
						))
					) : (
						<NoData />
					)}
				</>
			);
		} else if (activeButton === 'Статус ИИ-расчета данных') {
			const history = Object.values(arrayData).flat();
			return (
				<div className={styles.wrapper_statusProgress}>
					<div className={styles.block__history}>
						<h2 className={styles.title__history}>История</h2>
						{history.reverse().map(el => (
							<HistoryCard key={el.task_id} data={el} />
						))}
					</div>
					<div className={styles.block__statusProgress}>
						{progress_load && Number(progress_load) > 0 ? (
							<>
								<h3>
									<span>Файл:</span> {file_name?.file}
								</h3>
								<span>Запросы:</span>
								<p>{post.system_prompt}</p>
								<p>{post.text_prompt}</p>
								<ProgressBar />
							</>
						) : null}
					</div>
				</div>
			);
		} else if (activeButton === MOSINFORM_TAB) {
			return <MosinformArchive filterText={filterText} />;
		} else if (activeButton === QUEUE_TAB) {
			return <MlopsQueue filterText={filterText} />;
		}
	};

	const hasFiles = allData && allData.length !== 0;
	const isArchive = activeButton === MOSINFORM_TAB || activeButton === QUEUE_TAB;
	const styleContent = {
		justifyContent: hasFiles || isArchive ? '' : 'center',
		alignItems: hasFiles || isArchive ? '' : 'center',
		paddingTop:
			activeButton === 'three' ? 'calc(24/1440*100vw)' : 'calc(92/1440*100vw)',
		paddingRight: activeButton === 'three' ? '0px' : 'calc(44/1440*100vw)',
	};

	return (
		<div className={styles.wrapper_dataSet}>
			<PanelTargetGraph
				handleClick={onClick}
				dataButtons={dataSetButtons}
				activeButton={activeButton}
			/>

			{showDevModal && (
				<div className={styles.modalOverlay} onClick={() => setShowDevModal(false)}>
					<div className={styles.modalContent} onClick={(e) => e.stopPropagation()}>
						<button 
							className={styles.closeButton}
							onClick={() => setShowDevModal(false)}
						>
							×
						</button>
						<div className={styles.modalIcon}>🚧</div>
						<h2>Функция в разработке</h2>
						<p>Раздел "Файлы кластеризации авторов" находится в активной разработке</p>
						<div className={styles.features}>
							<h3>Планируемые возможности:</h3>
							<ul>
								<li>📊 Визуализация кластеров авторов</li>
								<li>📥 Экспорт для Embedding Projector</li>
								<li>👥 Детальная информация о каждом авторе</li>
								<li>🔍 Анализ связей между авторами</li>
							</ul>
						</div>
						<button 
							className={styles.okButton}
							onClick={() => setShowDevModal(false)}
						>
							Понятно
						</button>
					</div>
				</div>
			)}

			<div className={styles.block__content} style={styleContent}>
				{activeButton !== 'Статус ИИ-расчета данных' && (
					<div className={styles.block__field}>
						<img
							src='/images/icons/input_button/search.svg'
							alt='search'
							className={styles.image__search}
						/>
						<input
							type='text'
							className={styles.input__search}
							placeholder='Поиск по названию'
							value={filterText}
							onChange={handleInputChange}
						/>
					</div>
				)}
				{renderContent(activeButton, { values: filteredData })}
			</div>
		</div>
	);
};

export default DataSet;